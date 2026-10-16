# Periodic matrix verification

`PeriodicInitialization` is an encoded tensor repeat operation independent of
matrix multiplication. `PeriodicMatmulPlan` exploits a narrower structure:
within each contiguous K segment, A repeats in M and B repeats in N. Preparation
computes an F64 product for each pair of small effective operand tiles. Output
regions sum the corresponding entries of those products. The same immutable
initialization descriptions must generate the operands sent to the device.

The initial numerical contract covers real F32 accumulation and
`D = alpha * A * B + beta * C + bias` with F32 epilogue arithmetic/output.
C and bias may be nonperiodic. Each `checkRegion` call receives their matching
logical regions, including correctly sliced broadcast vectors. Transposition,
batching, device layout, upload, readback and launch ordering belong to the
consumer. No sampled check can establish that arbitrary arrays match a declared
periodic definition.

## Exact-integer mode

Preparation converts stored seed values to the effective F32 input values,
requires finite integers, and bounds all possible partial sums. Region checks
also require integral alpha, beta, C and bias, and bound epilogue intermediates.
Every absolute bound must be strictly less than 2^24. Within this contract,
products, arbitrary partial sums and epilogue operations are exactly
representable in F32. The expected F64 result is also exact, and comparison
uses zero tolerance. This is a sufficient eligibility condition; some exact
problems outside this bound are conservatively unsupported.

## Floating-point mode

Preparation uses the decoded **stored** seed encodings, including their F32
compute conversion. The reference is an F64 calculation of those effective
operands, rather than an emulation of a particular device reduction order.

For segment s, compute a conservative bound

```
W_s = max_i sum_k |A_s[i,k]| * max_j |B_s[k,j]|
W   = sum_s W_s
V   = |alpha| * W + |beta| * max|C| + max|bias|
```

These bounds require work proportional to the small inputs, and do not require
an additional matrix product. For every output, W bounds the sum of absolute
products, including every partial sum irrespective of cancellation. The
epilogue bound V bounds the corresponding sum of magnitudes. All bound products
and additions are rounded upward using `nextafter` after each F64 operation.

Let `gamma(n,u) = n*u / (1-n*u)`, with `u32=2^-24` and `u64=2^-53`. For K terms
and S nonempty segments, use

```
n32 = 2*K + 4
n64 = 2*K + 2*S + 4
tolerance = (gamma(n32,u32) + gamma(n64,u64)) * V
```

The bound deliberately allows separate product and sum rounding: no contribution
passes through more than 2*K+4 device arithmetic operations for the supported
linear expression, including the epilogue. The reference additionally combines
the segment results; 2*K+2*S+4 bounds its operation paths. Standard forward-error
bounds apply to each expression, and the triangle inequality bounds their
difference. Fused multiply-add and shallower summation trees also satisfy this
conservative bound. The implementation rounds gamma and the final tolerance
upward. It compares the observed F32 output directly with the F64 reference,
without adding another F32 rounding to the reference.

Both expressions must remain within the specified arithmetic model. Preparation
and checking reject nonfinite values, subnormal effective inputs/coefficients,
and overflow-risk bounds. They conservatively bound each nonzero F32 value's
dyadic quantum by `2^(ilogb(abs(x))-23)` and reject cases whose product or
epilogue quantum could fall below the minimum normal F32 value. This excludes
underflow and flush-to-zero ambiguity, including cancellation into subnormals.
They also reject K for which gamma is unbounded, unsupported input transforms,
low-precision accumulators, activations, and intermediate low-precision stores.
The caller must select device solutions that implement the declared arithmetic.

This tolerance is fixed from the inputs before considering the observed result.
It is an absolute per-element criterion. Cancellation tests deliberately use
non-integer values for which F32 accumulation is inexact; different legitimate
summation orders must pass, while finite errors beyond the bound must fail.

## Resource and result contracts

Preparation reports small-product multiply-add counts, seed/product storage,
and a preparation storage estimate. It checks work/storage limits before making
small products. Region size is checked before allocating expected outputs;
consumers control the readback block size. A budget failure is Incomplete,
unsupported numerical semantics are Unsupported, and a compared wrong result
is Mismatch. No outcome silently invokes a full reference GEMM.

Products repeat under the least common multiple of the segment row and column
periods. Larger coprime periods reduce these complete-product aliases but cannot
remove each segment's repeated input values. All outputs are compared for the
chosen inputs; this does not prove detection of all address faults. Ordinary
nonperiodic input families remain necessary in the wider test suite.
