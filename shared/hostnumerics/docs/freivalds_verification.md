# Exact-integer Freivalds verification

`prepareFreivalds` prepares a reusable numerical check of one logical rank-two
matmul and a linear epilogue. It needs ordinary eligible inputs, with no special
generation pattern. Products retain responsibility for device execution,
layouts, batching, output padding, and solution/iteration diagnostics.

Preparation owns compact encoded snapshots of A, B, the active C addend and
bias, together with combined row/column factors. Changing or releasing the
caller's buffers cannot change a prepared plan, including its exceptional-value
reference calculations. Plan copies share private immutable state. Inputs may
have strided or transposed layouts; their padding is not retained.

## Numerical contract

The supported expression is

```text
D = convert_output(scaleD * (alpha * diag(rowScale) * A * B
                            * diag(columnScale) + beta * C + bias)).
```

Here A and B are the effective operands after the ordered
`MatmulOptions::preQuantizationScalesA/B` and `computeTypeA/B` conversions.
Operand scales may broadcast along any axis, including K. Each scaling prefix
must remain integral and exactly represented by the accumulator type, and the
compute-input conversion must preserve its value. For example, converting 2049
from F32 to F16 is refused. This is conservative certification, not a model of
rounded operand transformations. Real-input conjugation is a no-op; complex
inputs, block scales and nondefault math modes are unsupported.

Use `MatmulOptions` for the operand transforms and accumulator type, and
`EpilogueOptions` for the linear terms. `inputScaleFactors` may be scalar, row,
or column factors. They
combine in their declared order before multiplication by the product. C and
bias may have any broadcast-compatible shape. Alpha, beta and the output scale
are rank-zero tensors. An absent addend ignores its coefficient, and beta zero
does not read C. Empty outputs read no operand values.

All effective values must be finite integers exactly represented by the
computation. Preparation validates conversion to the accumulator type, bounds
each prefix of the combined factors, and uses absolute sums to bound arbitrary
partial reductions. For example, for each row,

```text
rawBound[i] = sum_k |A[i,k]| * max_j |B[k,j]|.
```

Scaled products, pre-output-scale epilogue intermediates and final results are
bounded too. The bounds must be strictly below the compute type's consecutive
integer limit: `2^24` for F32, `2^53` for F64, `2^11` for F16, or the
corresponding supported integer/narrow-type limit. This is a conservative
eligibility rule. A cancellation or a later zero scale cannot hide an unsafe
earlier intermediate. These bounds also keep reference arithmetic within
int64 and the exact integer range of double.

Nonlinear operations and auxiliary checks are not part of this primitive.
A product can pass already normalized effective
inputs, or compose separate activation/E/amax/gradient checks. The hipBLASLt
adapter preserves those existing operations around its linear check.

## Projections and exceptions

Arithmetic uses the prime `P = 2^61 - 1`. For a nonzero right probe r and left
probe t, preparation computes the expected `D*r` and `t^T*D` from factored
matrix-vector products, incorporating the linear epilogue modulo P. It does not
compute a dense M-by-N reference product. The projection work is
`O(MK + KN + MN)` when C or bias needs full traversal.

`plan.copyRequest()` provides the shape, output encoding, integer probes, request
identity, fast-path magnitude threshold, and maximum exception count. A device
adapter must visit every logical output exactly once in each direction. Finite
integer stored values strictly below the threshold contribute to both sums.
Other values are omitted from both sums and recorded once, with their logical
coordinate and decoded stored value. Padding is outside this protocol.

`plan.check(observation)` checks the identity, dimensions, element count,
canonical residues, budgets, and unique in-range exception coordinates before
accepting the observation. Each exceptional element is recomputed exactly from
the immutable inputs, converted to the requested output encoding, and compared
to the stored value. Its unrounded integer reference then enters both sums.
Correct output rounding/saturation is supported, including an infinity when
that is the conversion's correct result. NaNs compare unequal. This work costs
`O(exceptionCount * K)` and can dominate the check for narrow outputs.

The adapter's completeness claim is trusted: host metadata cannot prove that a
device kernel actually scanned every element. Adapter tests must exercise that
contract. The HIP-free `plan.check(tensor)` is a reference implementation of the
same observation protocol and numerical check.

`Passed` means a complete observation agreed. `Mismatch` is a numerical failure.
`Unsupported` at preparation means that the operation lacks a certified
numerical contract. `Incomplete` reports an exhausted budget or malformed,
stale, or incomplete observation; it is never a passing fallback.

## Probability and reproducibility

For a fixed nonzero error matrix over the field and independent uniform probe
entries in `[1, P)`, a single direction misses the error with probability at
most `1/(P-1)`. A single changed element is detected because every probe entry
is nonzero. Error magnitudes in the certified fast range cannot alias to zero
modulo P. The second direction supplies column diagnostics as well as another
check.

The supplied SplitMix algorithms produce reproducible seeded vectors. They
are not independent random field samples, cryptographic challenges, or a
guarantee against errors chosen after seeing the probes. The default maps to
61 bits and rejects endpoints; this mapping is unbiased given independent
uniform source words. `SplitMix64Legacy` preserves hipBLASLt's established
modulo mapping and deterministic cases. Do not apply the ideal probability
bound as a proved guarantee for either seeded generator. Different seeds are
useful test coverage, rather than an adversarial-security mechanism.

## Resource limits

`FreivaldsLimits` bounds preparation storage, projection multiply-add work,
observation storage, exceptional recomputation and reported coordinates.
Preparation checks its storage estimate before shape-sized allocation, including
operand transformation scratch. Ordered scales also count toward its work limit.
`retainedInputBytes` counts only encoded input snapshots; factors, probes and
other summaries are additional storage covered by the preparation budget.

The observation budget covers vector payload capacities, working copies,
sorted exception coordinates and bounded diagnostics. It is checked before
copying projections or allocating the coordinate list. CPU exception growth is
capped before allocation, including vector growth capacity. An optional
per-check exception limit can further restrict work when several batches share
one budget. Zero K charges one work unit per exception so it cannot create an
unbounded list.

Storage bounds concern requested numerical payloads, excluding allocator
bookkeeping, thread stacks, runtime allocations, caller input storage, and
product-owned device buffers. They are not measured process peak memory.
`exactValue` and `roundedValue` are explicit single-coordinate operations;
callers must budget repeated diagnostic use themselves. HipBLASLt includes its
bounded diagnostics in the launch's recomputation budget.

Numerical probe and projection payloads are UInt64 Tensors. The plan exports
detached, contiguous copies with `copyRequest()`, `copyExpectedRowSums()` and
`copyExpectedColumnSums()`: Tensor constness is shallow, so sharing private
storage would allow a caller to corrupt future checks. The observation accepts
strided tensors and charges their complete backing storage, plus logical working
copies, against its storage budget before allocating those copies. Empty-output
plans use empty probes and projections. `reduceFreivaldsProjections()` combines
rank-two device partials over their chunk axis using modular integer addition.

## Supporting exact checks

`exact_verification.hpp` supplies three operations used alongside linear
verification. `verifyExactActivation` reconstructs an already verified auxiliary
result with division by its rank-zero scale, applies identity, ReLU or finite
clamp, scales and converts the output, and compares it with the observed Tensor.
It also returns the maximum magnitude before output scaling. Reconstruction uses
Float64 division; multiplying by a rounded reciprocal is not equivalent.

`recoverExactMaximumAbsolute` returns a rank-zero Tensor containing the maximum
of an already verified output, divided by the magnitude of its output scale.
Both operations reject nonfinite or fractional stored values, the floating-point
exact-integer boundary, integer saturation endpoints, and a zero or nonfinite
reconstruction scale. Verification of the original GEMM is a caller precondition;
these checks do not establish the source of the auxiliary or output values.

`verifyExactSum` independently certifies integral input values and a strict
absolute-sum bound for every reduction slice before using `referenceSum` and
`compare`. This is needed for a bias-gradient reduction even when the other GEMM
operand is zero and makes the GEMM bound vanish. It accepts arbitrary reduction
axes and strided layouts. Output conversion uses nearest-even integer rounding
and saturation explicitly. Failures outside the numerical contract are
`Unsupported`, with structured comparison evidence for supported checks.

The hipBLASLt adapter retains genuine Tensor storage owners for logical inputs,
expresses transposes through layouts, and passes decoded MX tensors directly.
CPU observations are explicit owned Tensors, separate from device descriptors.
The product still controls HIP transfers, physical offsets, padding and launch
diagnostics; numerical verification does not retain borrowed product pointers.
