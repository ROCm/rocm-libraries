# XF32 via BF16x3 MMA emulation

This document explains why rocKE provides emulated TF32 MMA atoms on gfx950, and how the
expansion works. It replaces the earlier solution plan and the step notes with one
description. It contains no measured performance results.

## Why rocKE needs it

The `tf32` dtype names the 19-bit float format that AMD hardware calls XF32. The format has
1 sign bit, 8 exponent bits, and 10 mantissa bits. An atom is one hardware
matrix-multiply-accumulate instruction on a small tile.

gfx942 registers two native TF32 atoms, `mfma_f32_16x16x8_xf32` and
`mfma_f32_32x32x4_xf32`. gfx950 has no native XF32 instruction and registered no TF32
atom, so a TF32 query there failed. Emulation means the software replaces an instruction
the hardware lacks. This work adds two emulated TF32 atoms to gfx950, so the same query
returns an atom on either target.

The emulation builds each TF32 product from three BF16 MMAs. gfx950 has fast native BF16
atoms, so the recipe gives the target the capability without new hardware. Each product
carries about 16 effective mantissa bits. That accuracy sits between TF32 with 10 bits and
FP32 with 24 bits, and it tracks the operand payload that the caller supplies.

One kernel body reaches either implementation. The MMA catalog is the per-arch table of
available atoms. A `(tf32, tf32, fp32)` query returns the native atom on gfx942 and the
emulated atom on gfx950, because both forms share one operand type and one catalog key.
Selection follows the capability of the target, not the request of the caller.

The recipe and its rounding match the packed path of Composable Kernel (CK), the public
reference implementation. CK ships that path with round-to-nearest-even (RNE) conversion
on gfx950, and rocKE targets the same bits.

## Catalog rows

An emulated row is a normal catalog row with one extra `emulation` descriptor. The
descriptor names the recipe and the physical atom, which is the hardware atom that the row
expands into.

```json
{
  "op_id": "mfma_f32_16x16x32_xf32",
  "family": "mma",
  "a": "tf32", "b": "tf32", "c": "fp32",
  "m": 16, "n": 16, "k": 32,
  "emulation": { "recipe": "bf16x3", "physical_op_id": "mfma_f32_16x16x32_bf16" }
}
```

Two rows ship, both on gfx950.

| Logical atom | Physical atom | Shape |
|---|---|---|
| `mfma_f32_16x16x32_xf32` | `mfma_f32_16x16x32_bf16` | 16x16x32 |
| `mfma_f32_32x32x16_xf32` | `mfma_f32_32x32x16_bf16` | 32x32x16 |

The rules below bind when the catalog loads.

- The `bf16x3` recipe fixes the operand dtype to `tf32`.
- The physical atom must be a native row of the same arch. A row that names an atom its
  target cannot issue claims support that the compiler later rejects, so the loader
  refuses it.
- The row inherits its fragment lengths, wave size, and layout maps from the physical
  atom. A fragment is an operand tile in the registers of one wave. The Python engine derives
  these values, and the static C++ table restates them, so a test on each side compares
  every value against the physical atom. A borrowed fragment cannot drift.
- A row with no `emulation` key is native. `MmaCatalog.enumerate` prefers a native atom
  over an emulated atom with the same `(m, n, k)`. No arch lists both for one `(m, n)`
  today, and a test pins that fact. An arch that mixes them must re-key the rule on
  `(m, n)` first.
- Absence of a row means explicit rejection. The engine never falls back from a TF32
  query to another dtype.
- An `(fp32, fp32, fp32)` query never returns an emulated atom. XF32 is opt-in.

gfx942 keeps its native atoms and ships no emulated row. The K-packed BF16 atoms that
these rows expand into do not exist on gfx942.

## The lowering

Lowering means translating the kernel IR into LLVM IR. The expansion sits at the
`tile.mma` lowering site in both engines, before ISA dispatch. The Python code lives in
`python/rocke/core/lower_llvm.py` and the C++ code lives in `cpp/core/lower_llvm/mma.cpp`.
The two engines emit byte-identical LLVM IR for every expansion. The site first asks
whether the `op_id` names an emulated atom on the arch being lowered. A native atom falls
through, so no existing kernel emission changed.

Each operand is a `vec<tf32xN>` vector carried as N I32 words, with N = 8 for both gfx950
atoms. The accumulator and the result are FP32 vectors, as with the native atoms. A
`bitcast` reinterprets each word as FP32 and never rounds it, so every payload bit reaches
the split.

The split writes a value as two BF16 parts. The expansion emits these lines per operand.

```llvm
%x       = bitcast <N x i32>    %tf      to <N x float>   ; payload, reinterpreted
%rne     = fptrunc <N x float>  %x       to <N x bfloat>  ; x rounded to nearest even
%bits    = bitcast <N x bfloat> %rne     to <N x i16>     ; ... as bit patterns
%sign    = and     <N x i16>    %bits, 0x8000             ; ... the sign bit
%cap     = or      <N x i16>    %sign, 0x7f7f             ; ... largest finite bf16
%over    = icmp ugt <N x i16>   %bits, %cap               ; ... did the rounding pass it?
%sat     = select  <N x i1> %over, <N x i16> %cap, <N x i16> %bits
%big     = bitcast <N x i16>    %sat     to <N x bfloat>
%big_f32 = fpext   <N x bfloat> %big     to <N x float>   ; ... widened back
%diff    = fsub    <N x float>  %x, %big_f32              ; ... what the payload lost
%small   = fptrunc <N x float>  %diff    to <N x bfloat>  ; ... rounded too
```

- `big` is the payload narrowed to BF16 with RNE, then held at the largest finite BF16
  with its sign.
- `small` is what the payload lost in that narrowing, and it is narrowed the same way. It
  comes from the payload, not from the capped copy.
- LLVM defines `fptrunc ... to bfloat` as RNE, and the AMDGPU backend selects the packed
  `v_cvt_pk_bf16_f32` instruction on gfx950. This lands on the CK packed path, so no
  truncation appears.
- The split emits as raw LLVM lines because the existing arith ops fix the scalar form,
  and a vector form does not exist yet.

The cap matters because RNE overflows to infinity for every payload of magnitude
`0x7f7f8000` or above, and that band holds the largest finite TF32 value. An uncapped
split then gives `big = +inf` and `small = -inf`. The `inf * 0` cross term turns a finite
product into NaN. Below the threshold the capped split is bit-identical to the uncapped
form. The cap also sets the non-finite behavior, with no special case. An infinity or NaN
operand makes every output element that consumes it non-finite, and makes no other output
non-finite.

The expansion then issues three MFMAs on the physical atom and threads one FP32
accumulator through them.

```
t0 = mfma(a_small, b_big,   C )
t1 = mfma(a_big,   b_small, t0)
D  = mfma(a_big,   b_big,   t1)
```

- The term order is fixed. FP32 accumulation is not associative, so another order
  produces different result bits, and a test pins the order.
- The expansion omits `a_small * b_small`. That term stays below `2^-16` of the product,
  which is the size of the error the split already carries. Operands exactly halfway
  between two BF16 values reach 99 percent of that bound.
- Only the last term writes the value that the caller named, so the expansion changes
  nothing downstream.

The emulation costs registers. Each operand arrives as eight I32 words instead of four
packed BF16 pairs, and the split materializes two more per operand. An inner loop often
reuses one A fragment across several N-tiles, so the atom re-splits the same fragment each
time. A later change can expose a split helper so the kernel splits once outside the loop.
That change is separate from this work.

The expansion exists only in the LLVM lowering. Both HIP lowerers reject an emulated atom
by name, rather than failing as a missing hardware instruction.

## Accuracy contract

- The split sees the whole payload, so the product tracks the exact product of the
  operands that the atom receives. On raw FP32 payloads that is more accurate than the
  native atom, which ignores the low 13 operand bits.
- On operands already rounded to TF32 precision, the native atom is exact and the
  emulation carries the omitted-term error, at most about `1.5e-5` relative per product.
- The two are not bit-reproducible against each other across targets.
- A payload in the top band splits against the cap and still sums to the exact product,
  and a product that truly overflows returns an infinity.
- An infinity against a partner of magnitude 512 or more can return NaN instead of the
  infinity. This is the one place where the result is weaker than IEEE, but it is still
  not finite.
- A subnormal payload no larger than `2^-134` (about 4.6e-41) contributes zero, because
  BF16 cannot hold its parts.

## Tests and gates

- `tests/core/test_xf32_bf16x3_lowering.py` reads the emitted IR back and re-derives every
  operand, and it hands the lowered kernel to LLVM's constant folder. The folded operands
  must equal the hand-derived bit patterns. A C++ twin pins the recipe on the C engine,
  and `tests/instances/parity/xf32_bf16x3_mma_emit.py` byte-compares the two engines on
  the expansion inside the byte-identity gate.
- `tests/core/test_xf32_comgr_compile.py` compiles each atom to a code object on the host,
  with no GPU, and reads the ISA. It expects exactly three `v_mfma_*_bf16` instructions
  and the packed `v_cvt_pk_bf16_f32` convert.
- `tests/core/test_xf32_selection_regression.py` asks two catalogs the same questions: one
  catalog holds the emulated rows, and the other drops them. The answers must match for
  every arch and every shipped atom, so nothing that predates XF32 moved.
- `tests/core/test_xf32_bf16x3_numerics.py` runs a numpy model of the recipe against
  FP32, TF32, and BF16 references. It normalizes error by `sum |a_i b_i|`, because a
  cancelling dot product must not hide or fake the arithmetic.
- `tests/core/test_xf32_bf16x3_device.py` runs
  `python/rocke/examples/gfx950/xf32/bf16x3_parity.py` on a gfx950 device. The harness
  compares the emulated atom bit for bit against a hand-written three-MFMA kernel. It also
  compares constant-operand products against closed-form values, with no layout model. It
  rejects injected faults, including a rotated term order and a dropped residual term.
- The byte-identity gate is GREEN for every family at llvm20, llvm22, and llvm23,
  including the `xf32_bf16x3_mma` parity family.

## Where the code lives

| Piece | Python | C++ |
|---|---|---|
| `tf32` dtype and carrier rules | `python/rocke/core/dtypes.py` | `cpp/core/dtypes.cpp` |
| Catalog schema and gfx950 rows | `python/rocke/core/arch/target.py`, `python/rocke/core/arch/data/arch_specs.json` | `cpp/core/arch/data.cpp`, `cpp/include/rocke/arch_target.h` |
| BF16x3 expansion | `python/rocke/core/lower_llvm.py` | `cpp/core/lower_llvm/mma.cpp` |
| Operand verifier | `python/rocke/core/tf32.py` | `cpp/include/rocke/tf32_internal.h` |


