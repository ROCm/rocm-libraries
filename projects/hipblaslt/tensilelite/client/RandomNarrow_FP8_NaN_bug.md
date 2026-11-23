# RandomNarrow Init Produces NaN for Float8/BFloat8 Types

## Bug

`rocm_random_narrow_range<Float8>` in `client/include/DataInitialization.hpp` used exponent range `[-100, 0]`, inherited from the float32 specialization. Float8 E4M3 OCP only has a 4-bit exponent (bias=7), so most of that range wraps around via unsigned arithmetic, frequently producing the all-ones exponent field that encodes NaN.

## Mechanism

The random generator (`rocm_random<Float8, -100, 0>`) works by:

1. Picking a random unbiased exponent `exp` in `[-100, 0]`
2. Computing `biased_exp = exp + expbias` (expbias=7 for E4M3)
3. Inserting the biased exponent into the Float8 bit pattern:
   ```
   u = signsig & ~expmask | ((exp + 7) << 3) & 0x78
   ```

The 4-bit exponent field occupies bits [3:6]. The expression `((exp + 7) << 3) & 0x78` extracts bits [0:3] of the biased exponent (mod 16). NaN in OCP E4M3 requires exponent field = 15 (0xF) **and** mantissa = 0b111.

### Which exponents wrap to field=15?

We need `(exp + 7) ≡ 15 (mod 16)`, i.e., `exp ≡ 8 (mod 16)`.

Values in `[-100, 0]` satisfying this: **-8, -24, -40, -56, -72, -88** → 6 out of 101 possible exponents.

### Per-element NaN probability

| Factor | Value |
|--------|-------|
| P(exp field = 15) | 6/101 |
| P(mantissa = 0b111) | 1/8 |
| **P(NaN per element)** | **6/808 ≈ 0.74%** |

### Impact on GEMM output

Each output element accumulates K inner-product terms. If **any** input element is NaN, the entire sum is NaN (IEEE 754: NaN propagates through arithmetic).

| K | P(at least one NaN per output element) |
|---|----------------------------------------|
| 128 | 1 − (1 − 6/808)^128 ≈ 62% |
| 256 | 1 − (1 − 6/808)^256 ≈ 86% |
| 512 | 1 − (1 − 6/808)^512 ≈ 98% |
| 1024 | 1 − (1 − 6/808)^1024 ≈ **99.95%** |

At K=1024, virtually every output element is NaN. Both GPU kernel and CPU reference produce NaN (they read the same poisoned input), so validation reports all mismatches (NaN ≠ NaN in IEEE 754).

## Symptom

```
[0]  elem=0 idx=0: -nan!=nan
Found 65536 incorrect values in 65536 total values compared.
```

Looks like a kernel bug (100% failure) but is actually a data init bug. The kernel passes with `init-a=One` or `init-a=Random`.

## Fix

Clamp the exponent range to the valid range for each sub-byte float type:

```cpp
// Before (broken): exponent range far exceeds 4-bit capacity
struct rocm_random_narrow_range<Float8> : rocm_random<Float8, -100, 0> {};

// After (fixed): [-7, 0] covers subnormals through 1.0
struct rocm_random_narrow_range<Float8> : rocm_random<Float8, -7, 0> {};
```

| Type | Exp bits | Bias | Old range | Fixed range |
|------|----------|------|-----------|-------------|
| Float8 (E4M3) | 4 | 7 | [-100, 0] | [-7, 0] |
| BFloat8 (E5M2) | 5 | 15 | [-100, 0] | [-15, 0] |
| Float8_fnuz (E4M3) | 4 | 7 | [-100, 0] | [-7, 0] |
| BFloat8_fnuz (E5M2) | 5 | 15 | [-100, 0] | [-15, 0] |

The existing code had a prescient comment: `// TODO: need to check for fp8 and bf8`.

## File

`tensilelite/client/include/DataInitialization.hpp`, around line 3099.
