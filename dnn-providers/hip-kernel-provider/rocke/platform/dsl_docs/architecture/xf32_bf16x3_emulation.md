# XF32 via BF16x3 MMA emulation

This document explains why rocKE provides emulated TF32 MMA atoms on gfx950, and how the
expansion works. It contains no measured performance results.

## Why rocKE needs it

The `tf32` dtype names the 19-bit float format that AMD hardware calls XF32. The format has
1 sign bit, 8 exponent bits, and 10 mantissa bits. An atom is one hardware
matrix-multiply-accumulate instruction on a small tile.

gfx942 registers two native TF32 atoms, `mfma_f32_16x16x8_xf32` and `mfma_f32_32x32x4_xf32`.
gfx950 has no native XF32 instruction, so a TF32 query there failed. Emulation means the
software replaces an instruction the hardware lacks. rocKE adds two emulated TF32 atoms to
gfx950. They build each TF32 product from three BF16 MMAs, on the native BF16 atoms that
gfx950 does have.

For ordinary-magnitude finite operands, products typically carry roughly 16 bits of
precision. This is not a full-domain relative-accuracy guarantee: subnormal payloads
have an absolute split-error bound of `2^-134` per operand, and magnitudes at or below
`2^-134` split to zero even when their product with a large partner is representable.
An infinity or NaN makes every output it feeds non-finite, but infinity against a
sufficiently large finite partner can become NaN rather than infinity; the emulation
is not IEEE-equivalent.

The MMA catalog is the per-arch table of available atoms. A `(tf32, tf32,
fp32)` query returns the native atom on gfx942 and the emulated atom on gfx950, so one
kernel body reaches either target. The recipe and its rounding match the packed path of
Composable Kernel (CK), the public reference implementation.

## Catalog rows

An emulated row is a catalog row with one extra `emulation` descriptor. The descriptor names
the recipe and the physical atom, which is the hardware atom that the row expands into.

```json
{
  "op_id": "mfma_f32_16x16x32_xf32",
  "family": "mma",
  "a": "tf32", "b": "tf32", "c": "fp32",
  "m": 16, "n": 16, "k": 32,
  "emulation": { "recipe": "bf16x3", "physical_op_id": "mfma_f32_16x16x32_bf16" }
}
```

Two rows ship, both on gfx950: `mfma_f32_16x16x32_xf32` over `mfma_f32_16x16x32_bf16`, and
`mfma_f32_32x32x16_xf32` over `mfma_f32_32x32x16_bf16`. The row inherits its fragment
lengths, wave size, and layout maps from the physical atom. A fragment is an operand tile
in the registers of one wave. Selection prefers a native atom over an emulated atom with
the same `(m, n, k)`. An `(fp32, fp32, fp32)` query never returns an emulated atom, and a
TF32 query on a target with no row fails explicitly.

## The expansion

Lowering means translating the kernel IR into LLVM IR. The expansion sits at the
`tile.mma` lowering site in both engines, before ISA dispatch, and the two engines emit
byte-identical LLVM IR. The Python code lives in `python/rocke/core/lower_llvm.py` and the
C++ code lives in `cpp/core/lower_llvm/mma.cpp`. A native atom falls through the same site
unchanged.

Each operand is a `vec<tf32x8>` vector carried as eight I32 words, and the accumulator and
the result are FP32 vectors, as with the native atoms. A `bitcast` reinterprets each word
as FP32 and never rounds it, so every payload bit reaches the split. The split writes a
value as two BF16 parts. The expansion emits these lines per operand.

```llvm
%x       = bitcast <N x i32>    %tf      to <N x float>   ; payload, reinterpreted
%rne     = fptrunc <N x float>  %x       to <N x bfloat>  ; x rounded to nearest even
%bits    = bitcast <N x bfloat> %rne     to <N x i16>     ; ... as bit patterns
%sign    = and     <N x i16>    %bits, 0x8000             ; ... the sign bit
%cap     = or      <N x i16>    %sign, 0x7f7f             ; ... largest finite bf16
%over    = icmp ugt <N x i16>   %bits, %cap               ; ... is the rounding past it?
%sat     = select  <N x i1> %over, <N x i16> %cap, <N x i16> %bits
%big     = bitcast <N x i16>    %sat     to <N x bfloat>
%big_f32 = fpext   <N x bfloat> %big     to <N x float>   ; ... widened back
%diff    = fsub    <N x float>  %x, %big_f32              ; ... what the payload lost
%small   = fptrunc <N x float>  %diff    to <N x bfloat>  ; ... rounded too
```

- `big` is the payload narrowed to BF16 with round-to-nearest-even (RNE), then held at the
  largest finite BF16 with its sign. `small` is what the payload lost, narrowed the same
  way, and it comes from the payload, not from the capped copy.
- The cap matters because RNE overflows to infinity for every payload of magnitude
  `0x7f7f8000` or above, and that band holds the largest finite TF32 value. An uncapped
  split gives `big = +inf` and `small = -inf`, and the `inf * 0` cross term turns a finite
  product into NaN. Below the threshold the split is bit-identical to the uncapped form.
- LLVM defines `fptrunc ... to bfloat` as RNE, and the AMDGPU backend selects the packed
  `v_cvt_pk_bf16_f32` instruction on gfx950. This lands on the CK packed path, so no
  truncation appears.

Then three MFMAs run on the physical atom, and one FP32 accumulator threads through them.

```
t0 = mfma(a_small, b_big,   C )
t1 = mfma(a_big,   b_small, t0)
D  = mfma(a_big,   b_big,   t1)
```

- The term order is fixed. FP32 accumulation is not associative, so another order produces
  different result bits, and a test pins the order.
- The expansion omits `a_small * b_small`. For ordinary-magnitude finite operands, that
  term stays below `2^-16` of the product, the same order as the split's representation
  error. Emulation can improve accuracy on raw FP32 payloads, but is not uniformly more
  accurate than native XF32. For TF32-prepared operands, native products can be exact
  while emulation retains the omitted-term error: for `A = B = 1 + 2^-8`, it gives
  `1.0078125` instead of `1.0078277587890625`. Native and emulated results are not
  bit-reproducible across targets.
- Only the last term writes the value that the caller named, so the expansion changes
  nothing downstream.

The emulation costs registers. Each operand arrives as eight I32 words instead of four
packed BF16 pairs, and the split adds two more per operand. The expansion exists only in
the LLVM lowering, and both HIP lowerers reject an emulated atom by name.

## How the work is gated

Tests pin the split lines and the term order on both engines, and LLVM's constant folder
re-derives the operands the MMAs receive. A compile gate builds each atom for gfx950 and
reads the ISA, which must show exactly three `v_mfma_*_bf16` instructions and the packed
convert. A differential test shows that every query that predates XF32 answers the same
with and without the emulated rows. On a gfx950 device, a harness compares the emulated
atom bit for bit against a hand-written three-MFMA kernel. The byte-identity gate is GREEN
for every family at llvm20, llvm22, and llvm23.
