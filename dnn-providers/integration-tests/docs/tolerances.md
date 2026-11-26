# Tolerances: how an engine's output is graded

Every output tensor of a bundle is compared element by element with its oracle
(golden data or a reference executor):

```text
|impl_i - ref_i| <= atol + rtol * |ref_i|      for every element i
```

One element outside the band fails the tensor. This document says where
`atol` and `rtol` come from, and explains the dynamic tolerance that grades
half, bfloat16 and float convolution outputs: forward, backward data (dgrad)
and backward weights (wrw). That tolerance is a statistical acceptance band,
not a bound: see "What kind of tolerance this is".

## Where the numbers come from

For each output tensor, the bundle harness takes the first of these that
applies:

| Source | Applies to | Where |
|---|---|---|
| A `[[validator_overrides]]` entry | The tensors its glob names | The engine's TOML (see [File Formats](file-formats.md)) |
| A `[[tolerance_overrides]]` entry | The tests its glob names | The engine's TOML |
| The convolution dynamic tolerance | half, bfloat16 and float outputs of a graph with one `ConvolutionFwd`, `ConvolutionBwd` or `ConvolutionWrw` node that accumulates in float, whose pointwise nodes and intermediates are half, bfloat16, float or double | `src/harness/tolerance/ConvDynamicTolerance.hpp` |
| The per-op default | Everything else; the loosest node of the graph wins, and the same value is used for `atol` and `rtol` | `TestTolerances.hpp` through `src/harness/tolerance/ToleranceResolver.hpp` |

TOML overrides describe how far one engine may drift. They are never applied
when golden data is checked against a reference executor, and neither is the
dynamic tolerance: only the engine's output is graded with it, and golden data
is checked against a reference executor with the per-op default.

## The convolution dynamic tolerance

The sections up to "Backward data and backward weights" are written for the
forward direction; backward data and backward weights use the same model with
the operands and `K` of their own direction and their own constants. The
constants are of two kinds, the SDK's defaults and an engine's own allowance
(see "SDK defaults and an engine's allowance").

Where this document names the conv-fwd test-plan bundles, it means those of
ROCm/rocm-libraries#14424 (the Variant3 to Variant6 graphs and the F-, L-, S-
and X-series cases). They are not in this tree; the case names and counts that
refer to them describe the data of a run on a merge of that branch with this one.

### Why a fixed number does not work

The per-op default for half and bfloat16 convolution forward is 0.1 for both
`atol` and `rtol`. At a typical output that is 17.5 % of the output at
`K` = 16 (`K` is the number of products summed per output), 11 % at 1,152
and about 10 % from `K` of about 1,000 on: 10 to 25 bfloat16 rounding steps,
and over 100 half steps.

The bound in the test SDK, `calculateConvFpropTolerance()`, scales with the
problem. Write `eps_c` for the epsilon of its compute type (2^-23 for float)
and `eps_out` for that of the output type. It multiplies the scale
`S = K * max|x| * max|w|`, the largest an output can be, by `gamma`, and when
the output type is coarser than the compute type it adds the output-cast term
`eps_out * S`. On its deterministic branch, which float compute takes up to
`K` of about 42,000, `gamma = 2 * K * eps_c / (1 - 2 * K * eps_c)`, used while
`2 * K * eps_c` is below 0.01: Higham's `gamma_K` with a factor 4 to spare
(`eps_c` = 2u). There it is a guarantee under its hypothesis, float
accumulation of the `K` products and one final rounding. It holds for any
data and any order of summing the products (Higham, Accuracy and Stability of
Numerical Algorithms, ch. 4; trees and blocks only shorten the chain), and it
needs no random inputs, no cancellation and no zero-mean rounding errors.
Beyond that, and for half and bfloat16 compute types at every practical `K`,
`gamma` is the 6-sigma statistical estimate `6 * sqrt(2K) * eps_c`, modelled on
the probabilistic analysis of Higham and Mary (2019; see "Where the summation
term comes from"). Winograd and intermediates kept below float are outside the
hypothesis: they are a different algorithm or a different arithmetic, not a
different order.

Its callers use the one number as both `atol` and `rtol`. Its growth with `K`
does not come from `gamma`: both terms multiply `S`, and for
`calculateConvFpropTolerance<T, T, float>()` the output-cast term
`eps_out * S` is 97 % of the bfloat16 number at `K` = 1,152 and 78 % of the
half one. With inputs of both signs a real output grows only as `sqrt(K)`, so
the band the bound makes, which grows as `K` relative to a typical output, is
wider than that output from `K` of about 100 on for bfloat16 (about 800 for
half), and passes almost anything.

The dynamic tolerance changes two things relative to that bound. The
magnitude is `sigma`, which grows as `sqrt(K)` because partial sums cancel,
instead of `S`; and the final rounding is priced relative to `|ref|` instead
of the largest possible output. The rounding statistics are not new. What is
new is the magnitude they are applied to, and with it an assumption about the
data: this is why the result is a statistical acceptance band and not a
bound (see "What kind of tolerance this is").

For float the per-op default is 1e-5 for both, not 0.1, and the comparison
goes the other way at long reductions: see "Float against the fixed 1e-5".

### The model

For an output stored as half or bfloat16, computed with float accumulation,
and graded against a float-accumulating reference (float adds a summation term
to `rtol` and takes a coefficient: see "Float outputs"), the statistical
acceptance band (see "What kind of tolerance this is") is:

```text
|impl_i - ref_i| <= 2 * eps * |ref_i|                      rtol
                  + sqrt(K) * u32 * sigma                   atol
                  + lambda * eps * M
                  + sum over rounding points of s * eps_p * m
                  + minNormal
```

| Symbol | Meaning |
|---|---|
| `K` | Products summed per output: the weight tensor's channel dimension (C / groups) times the filter volume (forward; see "Backward data and backward weights" for the other directions). |
| `sigma` | The typical output magnitude of zero-mean inputs, `sqrt(K * mean(x^2) * mean(w^2))`, times the factor by which the graded output carries the convolution's result: the product of the largest magnitudes of the tensors the fused multiplies apply, and the sum over paths where they meet again (see "What the harness feeds it"). The moments come from how the inputs were filled (see below). |
| `rms` | The root mean square of the output, `sqrt(K * mean(x^2) * mean(w^2) + K * (K - 1) * mean(x)^2 * mean(w)^2)` times the same factor: `sigma` plus what nonzero means of both inputs add (0 for zero-mean inputs). An engine's `lambda` and the rounding points are priced on it. |
| `M` | The magnitude `lambda` is priced on: `sigma` for the SDK's `lambda`, `rms` for an engine's (see "SDK defaults and an engine's allowance"). |
| `eps` | Machine epsilon of the output type: 2^-7 for bfloat16, 2^-10 for half. |
| `u32` | Unit roundoff of float, 2^-24. |
| `lambda` | The allowance for intermediates and partial results, in units of `eps * M`. The SDK's is for kernels that sum the `K` products in float and round once: forward 0 for bfloat16 and 3 for half (backward data and backward weights have their own, see "SDK defaults and an engine's allowance"). An engine's own allowance replaces it. |
| `minNormal` | The smallest normal number of the output type. |
| `eps_p`, `m`, `s` | One rounding point: the epsilon of the type rounded to, the root mean square of the value rounded, and the largest factor by which an error in it reaches the output. |

Each term is one way two correct implementations differ:

- **Output rounding** (`2 * eps * |ref_i|`). Both sides round their float
  result once to the output type. Two correct roundings of nearly equal values
  differ by at most one unit in the last place, and `eps * |ref_i|` is one to
  two of those units. The factor 2 is a margin over that bound, 2 to 4 units;
  the largest bfloat16 error in the calibration data was one rounding step.
- **Summation order** (`sqrt(K) * u32 * sigma`). Two float sums of the same `K`
  products in a different order differ by a random walk of rounding errors,
  about `u32 * sqrt(K) * sigma`. This is what decides an output near zero,
  where the relative term is small. See "Where the summation term comes from".
- **Intermediates in the io type** (`lambda * eps * M`). A kernel that sums
  the `K` products in float and rounds once still leaves a little beyond the
  final rounding (forward half: up to 1.98 `eps * sigma` for the non-split
  assembly implicit GEMM; bfloat16 none): the SDK's `lambda`. Kernels that are
  not that arithmetic need more: Winograd transforms, split-K kernels that add
  partial results in the io type, operands rounded below the io type. That is an
  engine's own allowance (see "SDK defaults and an engine's allowance").
- **Denormals** (`minNormal`). Hardware may flush denormal results to zero.
- **Roundings of intermediates** (the sum over the rounding points). A value
  rounded to half or bfloat16 before the output, which the reference does not
  round; see "Rounding points".

Why `lambda` is a constant, not a function of `K`: a fixed number of
half-precision roundings of intermediates the size of the output adds an
error proportional to `eps * sigma`, whatever `K` is. Summing in half
throughout would instead grow as `sqrt(K)`, and float accumulation of
half-rounded blocks is again constant. The measured half errors do shrink
somewhat at large `K`, but that follows which solvers MIOpen picks for which
shapes, not a rounding law, so the tolerance does not model it.
The SDK's `lambda` is flat in `K`. An engine's may grow with `K`: split-K
kernels that add partial results in the output type split a longer reduction
into more parts, and the allowance for them has a term in `K^(1/4)` (see "SDK
defaults and an engine's allowance").

`sigma` grows as `sqrt(K)` exactly as the output does, so the band is a fixed
fraction of the output at every reduction length. At a typical output
(`|ref|` = `sigma`) the band, divided by `sigma`, is `atol / sigma + rtol`.
For inputs uniform in [-1, 1] it is, for each way of setting the tolerance
(the SDK bound is used as both `atol` and `rtol`, as its callers do; the
dynamic tolerance at the SDK's defaults and at MIOpen's allowance, `lambda` of
12 for half):

| Output | K | fixed 0.1 | SDK bound | dynamic, SDK defaults | dynamic, MIOpen |
|---|---|---|---|---|---|
| bfloat16 | 16 | 0.175 | 0.22 | 0.016 | 0.016 |
| bfloat16 | 1,152 | 0.109 | 10.1 | 0.016 | 0.016 |
| bfloat16 | 65,536 | 0.101 | 535 | 0.016 | 0.016 |
| half | 16 | 0.175 | 0.027 | 0.0049 | 0.014 |
| half | 1,152 | 0.109 | 1.6 | 0.0049 | 0.014 |
| half | 65,536 | 0.101 | 82 | 0.0049 | 0.014 |

The SDK bound column prices `calculateConvFpropTolerance<T, T, float>()`, as
the sample passes it. With `double` compute, as the GPU reference tests pass
it, bfloat16 is 9.8 and 518 and half 1.2 and 65 at the two larger `K`.

The SDK bound is tighter than the dynamic tolerance only for half: at `K` of 6
or less against MIOpen's allowance, at `K` = 1 against the SDK's defaults.
Counted by its largest error against its `atol` alone, as the
first calibration did, it failed 22 of the 10,434 correct half and bfloat16
outputs graded (12 distinct cases); recounted at the current head, over more seeds,
it is 32 outputs of the same 12 cases, and used the way its callers use it (the
one number as both `atol` and `rtol`) 30 outputs of 10 cases. No case is a bound
that was not given the true input maxima: see "The cases the SDK bound failed".

The function is `calculateConvFpropOutputTolerance<T>()` in
`projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp`.

### SDK defaults and an engine's allowance

`lambda` has two sources, and a graded output gets one of them for its
direction and output type.

The SDK's defaults (`detail::convToleranceAllowances` in
`DynamicTolerancesConv.hpp`) price kernels that sum the `K` products in float
and round once, called conforming here. They were fitted on MIOpen's GEMM, CK,
naive, depthwise and non-split xdlops implicit-GEMM solvers, at a headroom of
at least 1.5 (data: "Which half solvers need lambda" for forward, "Calibration:
backward data and backward weights"). An engine gets them when it sets nothing,
and they are priced on `sigma`.

| Direction | float: `lambda`, `c` | half: `lambda`, `c` | bfloat16: `lambda`, `c` |
|---|---|---|---|
| forward | 0, 4 | 3, 1 | 0, 1 |
| backward data | 13, 4 | 6, 1 | 3, 1 |
| backward weights | 5, 4 | 11, 1 | 9, 1 |

Three kinds of kernel are not conforming, and what they leave is the engine's
allowance, not the SDK's:

- Winograd (`ConvBinWinograd*`, `ConvWinoRage*`): not a sum of the `K` products.
- Split-K kernels that add their partial results in the output type: the
  assembly backward-weights kernels for half (`_gkgs`, packed fp16 atomics), the
  CK grouped backward-weights kernels (2D and 3D), the CK grouped backward-data
  kernels (2D) and the assembly backward-data kernels for half (`_gkgs`).
- Kernels that round the operands below the io type: on gfx90a, MIOpen's
  backward fp16 alternate implementation rounds the operands to bfloat16.

An engine sets its allowance in its TOML, per direction and output type, next to
`[[tolerance_overrides]]`:

```toml
[conv_wrw_tolerance.half]
lambda = 13.0
lambda_per_quartic_root_k = 2.5

[conv_wrw_tolerance.half.gfx90a]
lambda = 35.0
lambda_per_quartic_root_k = 2.0
```

| Key | Meaning |
|---|---|
| `summation_order` | Float tables only: `c`, a finite positive number that replaces the SDK's 4 (see "The base coefficient and the engine's"). |
| `lambda` | A finite number, not negative, that replaces the SDK's `lambda` of the direction and type. |
| `lambda_per_quartic_root_k` | A finite number, not negative, only with `lambda`: it adds that times `K^(1/4)` to `lambda`, for kernels that split a longer reduction into more parts. Unset is 0. |

The tables are `[conv_fwd_tolerance.<type>]`, `[conv_dgrad_tolerance.<type>]`
and `[conv_wrw_tolerance.<type>]`, with `<type>` one of `float`, `half` and
`bfloat16`. A table named by an architecture under one of them
(`[conv_wrw_tolerance.half.gfx90a]`) holds the same keys for the devices whose
`gcnArchName` contains that name, and replaces the type table whole on them: a
key it leaves unset is the SDK's value, not the type table's. Another type or
key, a top-level table other than `[meta]`, `[[tolerance_overrides]]`,
`[[validator_overrides]]`, `[[test_skips]]` and the three above, and two arch
tables that match one device are errors (see [File Formats](file-formats.md)).
Only the bundle harness reads the tables. A `[[tolerance_overrides]]` entry that
matches a test still replaces the dynamic tolerance whole.

An engine value replaces the SDK's, also where it is lower: MIOpen's
backward-weights bfloat16 allowance, 1 + 2 `K^(1/4)`, is below the SDK's 9 for
`K` under 256. It is also priced differently. The SDK's own terms, the summation
term and its `lambda`, use `sigma`; an engine's `lambda` uses `rms`, means
included:

```text
lambda_engine(K) = lambda + lambda_per_quartic_root_k * K^(1/4)
atol = c * sqrt(K) * u32 * sigma + lambda_engine(K) * eps * rms
       + sum over rounding points of s * eps_p * m + minNormal
```

The mean term is the engine's because only the split-K kernels' error grows
with the partial sums, which with a nonzero mean grow with the output
(`K * mean * mean`), not as `sqrt(K)`; priced on `rms`, their allowance follows
it (the measurement is in "Backward summation coefficient against the theory
and float64"). The conforming arithmetic does not need it: its error is a
rounding of partial sums of the size of the output, which the `|ref|` term
covers (the two-step derivation of "Where the summation term comes from": the
`sigma` part is the zero-mean case and the `|ref|` part the mean).

MIOpen's allowances are in
`dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml`, each with its solver
family and ticket (ALMIOPEN-3243 for Winograd, ALMIOPEN-3380 for the split-K
kernels). Each keeps a headroom of at least 1.5 on the calibration data of both
architectures; where gfx90a needs more, an arch table holds its own value.

| Direction | float | half | bfloat16 |
|---|---|---|---|
| forward | `summation_order` 9 | `lambda` 12 | the SDK's |
| backward data | `lambda` 15 (gfx90a 18) | `lambda` 13 (gfx90a 46) | `lambda` 6 |
| backward weights | `lambda` 15 | `lambda` 13 + 2.5 `K^(1/4)` (gfx90a 35 + 2 `K^(1/4)`) | `lambda` 1 + 2 `K^(1/4)` |

The float tables of backward data and backward weights keep `c` = 4, and the
forward float table keeps the SDK's `lambda` of 0.

### What kind of tolerance this is

The dynamic tolerance is a statistical acceptance band, not a bound. A bound
asks whether an error is compatible with what the arithmetic permits, for any
data: the SDK function of "Why a fixed number does not work" is one on its
deterministic branch. The band asks whether an error is typical for an assumed
workload. An output outside it is a reason to look at the kernel; an output
inside it is not a proof that the kernel is right. It rests on:

- **The arithmetic.** The `K` products are accumulated in float, in some
  order, and the result is converted once to the output type. Roundings of
  intermediates below float are separate terms (`lambda`, the rounding points).
- **The rounding errors** (step 1 of "Where the summation term comes from").
  The relative error of each addition is independent of the others, has zero
  mean and a mean square of at most `u32^2 / 3`.
- **The products** (step 2 of the same section). They are independent, with
  variance `sigma^2 / K` and mean `ref / K`, so the partial sums are a random
  walk around a drift: where a product sits in the sum tells nothing about its
  sign or size. This is a separate assumption from the one on the rounding
  errors, and each fails on cases of its own (below).
- **Derived and fitted constants.** Derived: the factor 2 on `eps * |ref|` (a
  margin over two units in the last place), the form `sqrt(K) * u32` of the
  summation term with its split over `sigma` and `|ref|`, and the coefficient 1
  of the rounding points. Fitted: every `lambda` (the SDK's to the conforming
  solvers of MIOpen, an engine's to its own kernels; see "SDK defaults and an
  engine's allowance") and the float
  coefficient `c` (the value 1 of half and bfloat16 is a margin over a
  measured bfloat16 need of at most 0.43).
- **The probability it stands for.** A fixed `c` is a fixed number of standard
  deviations of the error at each element: at least `c / 0.41`, since 0.41 is
  the largest factor of the standard deviation (step 2), which is 9.8 for the
  base `c` = 4 and 22 for MIOpen's 9. The same number applies to every element,
  so the chance that some element of a correct output fails grows with the
  element count of the output. For a normal error the largest of 10^5 to 10^9
  elements reaches 4.5 to 6 standard deviations.

Each of the two assumptions has a case where it fails, and there the
classical bound holds and this band does not. A nonzero
mean alone is not one of them: the `|ref|` term covers it (with both inputs
uniform on [0, 1] the `c` needed is about 1, in emulation). The figures below
are from emulation on the host, not measured on an engine, and are not part of
the calibration tables.

- **Rounding errors that are not independent: constant fills.** With `x` and
  `w` both constant, every step of a sequential sum rounds the same way, and
  the error grows like `K * u32 / 4` instead of `sqrt(K)`. In float32
  emulation at `K` = 18,432 a sequential sum is off by up to 2.7e-4 relative,
  and a correct blocked or pairwise engine graded against the sequential
  reference needs `c` up to 34 (median 14). Even the step 1 estimate computed
  from the actual partial sums is too small here, by a factor 13 at `K` = 1,152
  and 52 at `K` = 18,432. Short repeating value lists show a milder version:
  8 % of them need more than 4.
- **Products whose sign follows the summation order.** With `x` nonnegative
  and the sign of `w` following the order (positive in the first half of the
  reduction, negative in the second), the partial sums reach about `K / 8`
  while `sigma` and `|ref|` are both of order `sqrt(K)`. The rounding errors
  behave as modelled but the magnitude of the band is wrong: in emulation a
  pairwise or blocked engine against the sequential reference needs `c` = 10
  at `K` = 1,152 and 36 to 41 at `K` = 18,432. Recipe fills cannot produce
  this. Golden inputs behind a ReLU, with a filter whose sign varies along the
  summation order, can.

When `x` and `w` are both filled with a constant, the per-op default is used.
Golden inputs and value lists are not detected, so the band applies to them.
Neither case is a regression: the fixed 1e-5 fails them too, and half and
bfloat16 are covered by `2 * eps`.

What supports using constants fitted to MIOpen for another engine: the SDK's
constants are the arithmetic of a conforming kernel, and MIOpen's extra lives
in MIOpen's file (see "SDK defaults and an engine's allowance"). The derivation
puts `c` at about 1.3 to 2.5 for two orderings of ideal float arithmetic, so the
base 4 is a margin over theory and does not depend on the engine, as long as
the engine accumulates in float and sums in some order. MIOpen's measured
non-Winograd solver families (1.77 to 2.59) sit there. Winograd's 5.68 is the
one number the derivation does not explain; MIOpen's file sets 9 for it (see
"The base coefficient and the engine's"). The SDK's `lambda` is measured on the
conforming solvers only, at a headroom of 1.5, so it prices a kernel that sums
the `K` products in float and rounds once. An engine whose kernels differ sets
its own `summation_order` and `lambda` in its TOML; a kernel that errs more than
its allowance needs a `[[tolerance_overrides]]` entry (see "When an engine
needs more").

### Where the summation term comes from

The float summation term has the form `sqrt(K) * u32 * (sigma + |ref|)`. It
comes from two steps that rest on different assumptions. `u32` is 2^-24.

Step 1, the rounding errors. A sequential float sum of the products `p_i`
satisfies `s_i = (s_(i-1) + p_i) * (1 + d_i)` with `|d_i| <= u32`, so to first
order the error is `sum_i d_i * s_i`, with `s_i` the exact partial sums
(Higham, ch. 4). If the `d_i` are independent with zero mean and
`E[d^2] <= u32^2 / 3`, then for any data:

```text
std(error | data) <= (u32 / sqrt(3)) * sqrt(sum_i s_i^2)
```

Step 2, the data. If the products are also independent, with variance
`sigma^2 / K` and mean `ref / K` (`sigma` is the quantity of the model: `sqrt(K)`
times the standard deviation of one product):

```text
E[s_i^2]    = i * sigma^2 / K + i^2 * ref^2 / K^2
std(error) <= u32 * sqrt(K) * sqrt(sigma^2 / 6 + ref^2 / 9)
           <= 0.41 * u32 * sqrt(K) * (sigma + |ref|)
```

The `sigma` part is the zero-mean case and the `|ref|` part is the mean, so a
nonzero mean is covered by construction. Each step has its own failure:
step 1 for constant fills, step 2 for products whose sign follows the
summation order (see "What kind of tolerance this is").

So `c` is about 0.41 times the number of standard deviations that the largest
of all elements reaches (4.5 to 6 for 10^5 to 10^9 elements, for a normal
error). That predicts a need of about 1.3 to 2.5 for two summation orderings
of ideal float arithmetic. The non-Winograd measurements (1.77 to 2.59, see
"Float outputs") sit there; `c` = 4 is a margin over the theory, and
Winograd's 5.68 is the one number it does not explain. The half and
bfloat16 bands keep `c` = 1 on the `sigma` term, and the bfloat16 data needs
at most 0.43 of it (see "Calibration"): below the figure the derivation gives
for ideal float arithmetic, and not explained by it.

References:

- Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM, 2002,
  ch. 4: `gamma_K` and the first-order expression of the error of a sum.
- Higham and Mary, A New Approach to Probabilistic Rounding Error Analysis,
  SIAM J. Sci. Comput. 41(5), A2815-A2835, 2019. It randomises only the
  rounding errors and keeps the scale `sum|p_i|`, so it supports the statistical
  branch of the SDK function and not `sigma`.
- Higham and Mary, Sharper Probabilistic Backward Error Analysis for Basic
  Linear Algebra Kernels with Random Data, SIAM J. Sci. Comput. 42(5),
  A3427-A3446, 2020. It takes both the rounding errors and the data as random.
  For data of zero mean the backward error is of order `u` whatever `n` is,
  which in forward terms is the `sqrt(K) * u * sigma` term; for a nonzero mean
  it returns to `sqrt(n) * u`, the `sqrt(K) * u * |ref|` term.

### Rounding points: intermediates below float

The model rounds once, the final result. A fused graph can round a value to
half or bfloat16 earlier, and an oracle that does not follow it. In develop's
Variant2 bundles (convolution, bias add, relu) the convolution's result is a
float virtual tensor and the add computes in the case's type through its
`compute_data_type`: the engine converts the result to that type before the
add and rounds the sum again, while the CPU reference adds in double and
rounds once. On the gfx942 data the engine's output matched an emulation with
those roundings at every element of the 144 bfloat16 outputs; the CPU
reference matched the emulation without them at 99.96 % or more of the
elements of each output. A rounded value that a later node cancels leaves a
difference no relative term of the output sees: bfloat16 results 1 and
1.0078125 differ by one step, and after adding -1 they are 0 and 0.0078125.

The CPU reference used to ignore the node's compute type: its pointwise nodes
added in double and rounded once, whatever `compute_data_type` said, while
the GPU reference and the engine honour it. That is fixed (#15139: the plan
passes the compute type to the reference), and the
two figures above were measured before it. Re-measured against the fixed
reference (see "Calibration of the rounding points"), the reference matches the
emulation with the roundings at 99.96 % or more of the elements of every
bfloat16 output (before the fix it matched the one without them), and the plain
band, without the terms of a node's own compute-type roundings, passes every
output of develop's Variant2 bundles but fails 28 of the 324 bfloat16 outputs
of the test-plan ones: a one-step difference in the convolution's float sum
flips the conversion of its result to bfloat16, and the add of the bias cancels
the result and leaves the step. No reference fix removes that, so the terms
stay, as do the points at stored tensors; the data needs at most 0.049 of the
compute-type term.

Every place between the convolution and the output where a value is rounded
to half or bfloat16 is a rounding point:

- a tensor one node writes and another reads, stored as half or bfloat16
  (virtual or not): it is rounded to its own type;
- an operand of a pointwise node that computes in half or bfloat16, when the
  operand is stored as another type: the node converts it, and rounds;
- the result of such a node, when it is stored as another type than the node
  computes in.

A graph output that no node reads is rounded once, to its type: that is
`rtol`, not a point. The exception is the result of a node that computes in half
or bfloat16 and stores it in another type: it is rounded in the compute type
first, and that rounding is a point whether or not a node reads the result.
Float and double intermediates add no point.

A value `v` rounded to a type with epsilon `eps_p` moves by at most half a unit
in the last place, `eps_p / 2 * |v|`. The elements of a tensor reach about
twice their root mean square `m` at a typical extreme (a 2-sigma level, not
the largest value), and there the rounding error is `eps_p / 2 * 2m`, that
is `eps_p * m`. One point contributes that level at its own place. It is not a
bound: an element beyond `2m` has a larger error. The error reaches the output
multiplied by the slope
`s` of the nodes in between: 1 through `ADD`, `SUBTRACT` and the modes assumed
to have a slope of about 1, and the largest magnitude of the scale through
a `MUL`. The error in the scale operand of a `MUL` that is itself rounded
reaches the output multiplied by the size `m` of the convolution's side of
the multiply. The slope is the one to the output being graded: a tensor with
no path to it is no point of it, and a graph with several outputs has its own
points and scale for each. Where the results meet
again, one error travels both paths and the contributions add: for
`ADD(MUL(y, s), y)` with `max|s|` = 3 the slope of `y` is 3 + 1 = 4.
The points add (they do not combine as a root of squares: with five points
the sum is at most 2.2 times larger, and the data does not tell the two apart):

```text
atol += sum over the points of  s * eps_p * m
```

The sum of the points is a statistical band like the rest of the tolerance: it
adds the typical-extreme levels of the roundings, and is not a bound on the
largest error they can make together.

The magnitude `m` follows the graph from the convolution's `rms` (without
the scales; `sigma` for zero-mean inputs). A value is a factor times the
convolution's result plus a
remainder that does not depend on it. A `MUL` multiplies both by the largest
magnitude of the scale. An `ADD`, a `SUBTRACT` and every other mode with
several operands add the factors (two paths from the convolution carry the
same error, so they add) and the mean squares of the remainders (independent
operands add in quadrature; the root mean square of a graph input comes from
its recipe, as for the convolution): `sqrt(a^2 + b^2)` for an operand from
the convolution's side and an independent one. A mode with one operand leaves
the value as it is. The magnitude of an operand is worked out only when a
point needs it, so no input is looked up, or scanned on the host, for a graph
without one.
The coefficient 1 comes from that level and is not fitted: the data needs at
most 0.115 of the term (see "Calibration of the rounding points").

### What the harness feeds it

`convErrorScale()` is asked for one output of the graph, the one being
graded, and applies the model only when the graph has exactly one
convolution node (`ConvolutionFwd`, `ConvolutionBwd` or `ConvolutionWrw`),
with float compute type, whose two operands (`x` and `w` forward, `dy` and
`w` backward data, `x` and `dy` backward weights) are graph inputs (not virtual, and not written by another node), and nothing else but
pointwise nodes: a node of another kind leaves the model. Every pointwise node
must compute in half, bfloat16, float or double, and every intermediate must
be stored as one of these: a tensor that one node writes and another reads
(the convolution's result, the pointwise results), virtual or not; an unset
type is none of them. A result that no node reads is a graph output and is
rounded once, as the model assumes, unless a node computing below float
stored it in another type (see "Rounding points"). A value stored or computed
in half or bfloat16 is a rounding point (see "Rounding points"); float and
double add none.

The coefficients an engine sets come from its TOML: for the direction of the
convolution and the type of the output being graded, the table
`[conv_<direction>_tolerance.<type>]`, or the table of the architecture nested
under it that the device's `gcnArchName` contains. The harness passes them to
the SDK function of the direction. A graph of an engine without such a table,
or whose table leaves a key unset, gets the SDK's value of that key (see "SDK
defaults and an engine's allowance").

The scale of an output is the factor by which it carries the convolution's
result. A `MUL` node with one operand from the convolution and the other a
graph input (in either operand order) multiplies the factor by that input's
largest magnitude; an `ADD`, a `SUBTRACT` and every other mode with several
operands add the factors, so `ADD(MUL(y, 0.5), y)` is `1.5 y`; a mode with one
operand leaves it as it is. A `MUL` on the branch of another output does not
enter it. A `MUL` of anything else (the result with itself, two inputs) leaves
the model, and so does an output that no node writes or that does not depend
on the convolution.

A pointwise node is modeled only for the modes whose slope is about 1: `ADD`,
`SUBTRACT`, `BINARY_SELECT` (which passes one operand on), `MUL` and the
activations `RELU_FWD` (plain, with an upper or a lower clip, or leaky with a
slope of at most 1), `IDENTITY`, `SIGMOID_FWD`, `TANH_FWD`, `SOFTPLUS_FWD`,
`ELU_FWD` (alpha at most 1), `SWISH_FWD`, `GELU_FWD` and
`GELU_APPROX_TANH_FWD`, and behind a dgrad `RELU_BWD`, `SIGMOID_BWD` and
`TANH_BWD` (the backward modes that have a reference, with the same limit on
the leaky slope), which multiply the gradient by the activation's derivative at
the forward input. Any other mode (`DIV`, `RECIPROCAL`, `SQRT`, ..., and the
five other backward activations, `SOFTPLUS_BWD`, `ELU_BWD`, `SWISH_BWD`,
`GELU_BWD` and `GELU_APPROX_TANH_BWD`, which no reference implements) and
a parameter that steepens an activation (a leaky slope or an elu alpha above
1) leave the model: the graph uses the per-op default. The conv-fwd test-plan
bundles the calibration used have `ADD` and `MUL`, and the activations above:
506 of their 1,132 half and bfloat16 `ConvolutionFwdPointwise` cases use an
activation other than `RELU_FWD`. At the parameters those bundles set
(softplus and swish beta 1, elu alpha 1) the largest slope among them is about
1.13 (gelu).

A graph whose `x` and `w` are both filled with one value (a `FIXED` recipe, or a
value baked into the graph) also uses the per-op default: see "What kind of
tolerance this is". In the test-plan bundles 12 of the 1,546 float
`ConvolutionFwd*` cases (the float32 `F21const` cases of `ConvolutionFwd`: 9 at
the full level, 2 comprehensive, 1 standard) fill both inputs with a constant,
read from the fill recipes of their sweeps. 28 float cases have a `FIXED` fill
on at least one input, none uses a value list, and none declares golden data.
A graph outside what is described here, an input whose statistics are not
available (a NaN or an infinity in golden data or in a fill recipe, an
unreadable baked value) use the per-op default.

A graph input's statistics come from, in this order:

- the value baked into the graph, for a scalar constant (in the test-plan
  bundles 40 of the 48 Variant3 cases, 27 of the 33 in half and bfloat16, bake
  their scale as 0.5; the `F27runtime` and `F27channel` cases fill it
  instead): `v^2`, `|v|` and `v`;
- the fill recipe, for a tensor the harness fills: `(lo^2 + lo*hi + hi^2) / 3`
  and `(lo + hi) / 2` for a uniform draw on `[lo, hi]`, `v^2` and `v` for a
  fixed value, the mean square and mean of the list for a value list, so no
  tensor is read;
- a scan on the host, for an input loaded from golden blobs, which is
  already there.

With zero-mean inputs partial sums cancel and the output grows as `sqrt(K)`.
With a nonzero mean on both operands (uniform on `[0, a]`, say; a constant
fill is a different case) they do not: the output grows as
`K * mean(x) * mean(w)`, which `rms` carries and `sigma` does not. `sigma`
would be 3, 25 and 190 times smaller than `rms` for uniform on `[0, 1]` at
`K` = 16, 1,152 and 65,536. The SDK's own terms use `sigma`: the `|ref|` term
covers the mean (see "Where the summation term comes from"). An engine's
`lambda` is priced on `rms`, because the kernels it allows for add partial
results in the output type, and those partial sums grow with the output (see
"SDK defaults and an engine's allowance"). A nonzero mean on one operand only
adds nothing: the other operand's signs still cancel. The calibration covered
the fills of the test-plan bundles, which are mostly zero-mean.

### Where the tolerance can exceed the output

`sigma` is the size of the convolution's output. An activation after it can
bound the graph's output, and then a half `atol` sized by `sigma` can be larger
than the range of that output, so no value in the range fails it. bfloat16 is
not affected: its `atol` stays below 0.011 in the same cases.

Recomputed from the sweeps of the test-plan bundles (nominal fill moments, the
formula above), the 268 half cases with a bounded activation (sigmoid, tanh,
relu clipped at 6, relu clipped to [0.1, 0.5]) have 43 with an `atol` above
the fixed 0.1 that the dynamic tolerance replaces. In the sigmoid
case `L19_1d_i1024_n1c64k64_f1024_fp16_cl_sigmoid` of the full level
(`K` = 65,536, `sigma` about 85) the `atol` is 1.0014 against an output in
[0, 1], so that case cannot fail on its values. For 17 of the 68 half sigmoid
cases the `atol` is at least a tenth of the range, and the largest half tanh
`atol` is 1.75 against a range of 2. The formula has no term for the range of
the graph's output, so it does not cap `atol` at it. Treat a half pass behind
a bounded activation as weak evidence for these long reductions.

### Calibration

Data: every half and bfloat16 `ConvolutionFwd` and `ConvolutionFwdPointwise`
case of the conv-fwd test-plan bundles that MIOpen runs, with outputs up to 4 M
elements. Each output was
compared with the harness reference and with a float64 recomputation from the
same inputs.

- The fit used 3 input seeds on an MI308X (gfx942), 5,091 correct outputs.
  Outputs that fail even the fixed 0.1 are MIOpen defects, not data for the
  fit, and were left out (7 cases).
- bfloat16: after the relative term, what is left is float accumulation
  noise, at most 0.43 `u32 * sqrt(K) * sigma`, flat in `K`. The largest error
  of any bfloat16 output is within the relative term: one rounding step.
- half: MIOpen's Winograd solvers and one assembly implicit-GEMM solver leave up
  to 8.1 `eps * sigma` on outputs near zero, mostly short reductions (table in
  "Which half solvers need lambda"). MIOpen's `lambda` of 12 is 1.48 times the
  largest value seen, 8.09. The SDK's 3 is 1.5 times 1.98, the largest need of a
  conforming solver, the non-split assembly implicit GEMM (the other conforming
  families need none).

The runs and tables below were made with `lambda` = 12 for half, which is
MIOpen's allowance (its file sets 12); the SDK's 3 follows from the per-class
needs of "Which half solvers need lambda".

Validation on data the fit did not see:

| Data | Fail the dynamic tolerance | Smallest headroom |
|---|---|---|
| gfx942, current head: 5 seed sets of the test-plan bundles, 4 of develop's | 0 | 1.437 (half), 2.000 (bfloat16) |
| gfx90a (MI250), current head: 3 seed sets of the test-plan bundles, 2 of develop's | 0 | 1.747 (half), 2.000 (bfloat16) |

Headroom is the tolerance divided by the error, at the element of an output
where that ratio is smallest: the factor by which the error could grow before
the output fails. It is 1 at the point of failure. The headrooms of this
document are the minimum over every element of every dumped output (outputs up
to 2^27 elements; the failure counts use every element too). The first
calibration took it over 48 largest-error and 48 random elements per output and
reported 1.44 and 2.0, upper bounds; the exact values are 1.437 and, on gfx90a,
1.747. The smallest bfloat16 headroom is exactly 2: one rounding step at a power
of two is half of the `2 * eps * |ref|` term.

The C++ implementation was then run through the harness, on every half and
bfloat16 conv-fwd case MIOpen runs, at all four levels. The MI325X runs and the
first three MI250 runs scanned the input tensors for their statistics. The
recipe path that ships (see "What the harness feeds it") ran on a fourth
MI250 run, which repeats one of its seed sets, and on the final revision on an
MI308X:

| Hardware | Input seed sets | Statistics from | Failures the dynamic tolerance adds |
|---|---|---|---|
| MI325X (gfx942) | 4 | tensor scan | 0 |
| MI250 (gfx90a) | 3, and a fourth run repeating one | tensor scan; the repeat from the recipe | 0 |
| MI308X (gfx942) | 1 (the bundles' own seeds) | recipe, final revision | 0 |

The MI308X run (filter `*_ConvolutionFwd*.*_bf16_*:*_ConvolutionFwd*.*_fp16_*-*.X3_*`,
MIOpen engine) selected 2,826 cases: 1,803 passed, 7 failed and 1,016 were
skipped (848 the engine does not rank, 168 quarantined in the TOML).

The fused convolution graphs `ConvolutionFwdPointwise` Variant3 to Variant6
(an activation, or a bias and an activation, after the convolution; only
Variant3 multiplies, by the scale) have no engine run in this table: the
MIOpen engine ranked no engine for any of their half and bfloat16 cases, so
the 0.5 scale of Variant3 has been graded only in host unit tests with mocked
engine output. The first engine and architecture that executes them should
add the headroom of those outputs here. Variant2 (a bias add, with or without
a relu) runs on gfx942 through the fused CK solvers; its calibration is under
"Calibration of the rounding points".

On gfx942 every failure in those runs is an output MIOpen computes wrongly
(values with the wrong sign), which also fails the fixed 0.1: the 3D
channels-last `F16fast` and `F16chan` cases of the comprehensive and full
levels. On gfx90a the retained runs also hold execute errors (56 of the
`F20` mixed-type cases with a float32 output, and one 3D channels-last
`F16fast` bfloat16 case, which return a MIOpen error status and compare
nothing) and wrong values in 3D and mixed-layout cases. One correct gfx90a
half output that the fixed 0.1 failed (one element 0.12 off at `|ref|` of
about 0.1, with `K` = 18,432) passes.

How much tighter it is: the median headroom per output, by range of `K` (up to
64, 65 to 512, 513 to 4,608, above), is 2.3, 2.1, 2.0 and 2.7 for bfloat16 and
12.3, 9.4, 6.3 and 19.1 for half. For the fixed 0.1 it is 48, 19, 15 and 21
(bfloat16) and 163, 119, 106 and 125 (half), and for the older bound with the
true input maxima 100 to 53,000 (bfloat16) and 42 to 107,000 (half).

Against the fixed 0.1 per element (a band of 0.1 + 0.1 * `|ref|`), bfloat16 is
tighter at every element and every `K`: for inputs uniform in [-1, 1] its
`atol` is 0.0013 at `K` = 65,536 and its `rtol` is 0.0156. half is tighter
relative to a typical output at every `K` (1.4 % of it at MIOpen's `lambda` of
12, 0.5 % at the SDK's 3, against 10 % to 17.5 %), but not for an element near
zero, where the band is the `atol`. The `atol` grows with `sigma`, which is
`sqrt(K / 9)` for those inputs. At 12 it passes 0.1 from `K` of about 650 on:
0.133 at `K` = 1,152 and 0.53 at 18,432 (at 3: from about 10,500 on, 0.033 and
0.133). At `K` = 18,432 (`sigma` about 45) the dynamic band is wider than the
fixed one for outputs smaller than about 4.4 in magnitude (0.33 at 3). That is
the intended
behavior and it cannot be capped at 0.1: the error of the half kernels in
absolute terms grows with the size of the outputs, and the correct gfx90a
output above, 0.12 off at `K` = 18,432, is already outside the fixed 0.1.

Detection, measured by injecting each bug into the exact result and grading
it against the real reference (gfx942, 3 seeds; half at MIOpen's `lambda` of 12):

| Injected bug | bfloat16: fixed 0.1 | bfloat16: dynamic | half: fixed 0.1 | half: dynamic |
|---|---|---|---|---|
| One filter tap missing | 95 % | 97 % | 95 % | 97 % |
| One input channel missing | 96 % | 98 % | 96 % | 99 % |
| Accumulator in the io type | 34 % | 97 % | 1 % | 5 % |
| One element off by 4 rounding steps | 0 % | 96 % | 0 % | 1 % |

No per-element tolerance detects an output truncated instead of rounded:
that error is half a rounding step. half cannot tell an io-type accumulator
apart from the kernels the `lambda` term allows for. For half the absolute
term (`12 * eps * sigma`) passes the fixed 0.1 from `K` of about 660 on for
inputs uniform in [-1, 1], and is about 0.5 at `K` = 16,000 (with the SDK's 3:
from about 10,500 on, and about 0.12 at 16,000). A missing filter
tap is `sqrt(C)` products against an output of `sqrt(C * R * S)` of them, so
how much of the half `atol` it fills depends on the filter volume `R * S`, not
on `K` alone: its ratio to the `lambda` term is `1 / (lambda * eps * sqrt(R * S))`
for uniform fills, above 1 only while `R * S` is below about 7,000 at 12 and
117,000 at 3 (computed,
not measured). Among the 21 calibration outputs of half with `K` above 16,000
(their filter shapes are not listed here), a missing filter tap (the products
of one filter position over all input channels) is caught in 29 % against
62 % for the fixed 0.1; a missing input channel is still caught in all of
them.

### Calibration of the rounding points

Data: every `ConvolutionFwd*` case of develop's conv-fwd bundles that MIOpen
runs (all levels, the X3 cases left out), and the same with the test-plan
bundles merged in, on gfx942 (the bundles' own seeds and 2 reseeded sets) and
gfx90a (the own seeds and 1 reseeded set). Every output was compared with the
harness reference and with a float64 recomputation from the same inputs, with
the rounding points recomputed from the graph; outputs that fail even the
fixed 0.1 are MIOpen defects and left out. Only the Variant2 graphs (bias add,
then relu) have rounding points: on gfx942 their add computes in the case's
type over float tensors, through the fused CK solvers. No case with a point ran
on gfx90a.

| Output | Data | Outputs with points | Fail without the term | Fail with it | Smallest headroom | Coefficient the data needs |
|---|---|---|---|---|---|---|
| bfloat16 | develop bundles, 3 seed sets | 144 per set | 110, 126, 105 | 0 | 2.60 | at most 0.096 |
| bfloat16 | test-plan bundles, 3 seed sets | 62 per set | 49, 49, 47 | 0 | 2.41 | at most 0.115 |
| half | develop bundles, 3 seed sets | 144 per set | 0 | 0 | 4.92 | none |
| half | test-plan bundles, 3 seed sets | 65 per set | 0 | 0 | 3.03 | none |

The coefficient the data needs is the largest error left after the other terms,
over the term itself, per output. It is at most 0.115 over every output with
points, so the coefficient 1, which comes from the bound above, has a margin of
about 9 over what the data needs. The term is for bfloat16: half passed without
it, and its headroom is 14 % larger with it.

Detection, on the 432 bfloat16 and 432 half outputs with points of develop's
bundles (3 seed sets; counts of outputs detected, out of 432 unless stated),
injecting each bug into the exact result and grading it against the real
reference. The fixed 0.1 is what such a graph was graded with before the model
covered it:

| Injected bug | bfloat16: fixed 0.1 | bfloat16: dynamic | half: fixed 0.1 | half: dynamic |
|---|---|---|---|---|
| One filter tap missing | 388 | 392 | 388 | 428 of 430 |
| One input channel missing | 432 | 432 | 432 | 432 |
| Bias missing | 432 | 432 | 432 | 432 |
| Accumulator in the io type | 0 | 46 of 430 | 0 | 0 |
| One element off by 4 rounding steps | 0 | 182 | 0 | 0 |

On the test-plan bundles, outputs with points are graded a little looser
than the fixed 0.1 against the largest bugs: of 186 bfloat16 outputs, an
input shifted is detected in 183 (the fixed 0.1: 186), a missing channel in
168 of 171 (171 of 171).

The C++ implementation then ran through the harness on the same cases (the
bundles' own seeds and the reseeded sets) and added no failure to the earlier
head on either architecture:

| Bundles | Hardware | Seed sets | Passed | Failed |
|---|---|---|---|---|
| develop's | gfx942 | 3 | 1,003 of 1,219 each (216 skipped: no engine ranks them) | 0 |
| develop's | gfx90a | 2 | 138 of 1,219 each | 0 |
| with the test-plan bundles | gfx942 | 3 | 2,849 to 2,850 of 4,739 | 20 to 21, the same cases as before |
| with the test-plan bundles | gfx90a | 2 | 2,108 to 2,109 of 4,739 | 8 to 9, the same cases as before |

All 144 bfloat16 and 144 half Variant2 cases of develop's bundles that the
engine ranks pass on gfx942 under the model, among them the 4 quick and 28
standard ones; they are graded against the CPU reference, which does not apply the
add's compute type, while the engine does (see "Rounding points"). The failures
that remain are outputs MIOpen computes wrongly: the 3D channels-last `F16fast`
and `F16chan` cases, and, on gfx942, the half Variant2 cases whose intermediates
are half (`F25first`, `F25full`, `F25scalar`), which also fail the fixed
0.1.

### Float outputs

A float output is graded with the same model (ALMIOPEN-3249), under the same
conditions as half and bfloat16. The forward output has no `lambda` term (the
backward ones have, see below), and the order
of the float summation sets both bands: summation error grows as
`c * sqrt(K) * u32` times the size of the partial sums, which are about
`sigma` for an output near zero and about `|ref|` or more for a large one, so

```text
|impl_i - ref_i| <= (2 * eps + c * sqrt(K) * u32) * |ref_i|     rtol
                  + c * sqrt(K) * u32 * sigma                   atol
                  + the rounding points + minNormal
```

with `eps` = 2^-23. The half and bfloat16 bands are unchanged: their `c` is 1,
and the relative summation term, under 1 % of their `rtol`, is left out.

`c` is not a constant of the model: how large the summation-order error is
depends on the solver MIOpen picks, and the tolerance is computed before the
engine runs. In the calibration data (the float `ConvolutionFwd` and
`ConvolutionFwdPointwise` cases of develop's and the test-plan bundles, on gfx942
and gfx90a, 6 input seed sets on gfx942 and 5 on gfx90a; outputs MIOpen computes
wrongly (an error of 0.1 or more, which no band of this section passes) and the
denormal cases left out) the
solver is in MIOpen's log. Per output, the `c` it needs is the largest error
left after
`2 * eps * |ref|`, over `sqrt(K) * u32 * (sigma + |ref|)`. The largest, per
solver family, over the outputs whose test ran that one family (the non-zero-mean
cases, which need the relative term, are in the data and do not change these
maxima):

| Solver family | Outputs | Largest `c` needed |
|---|---|---|
| Winograd (`ConvBinWinograd*`, `ConvWinoRage*`) | 1,178 | gfx942 5.68, gfx90a 5.40 |
| CK fused (`ConvCKIgemmGrpFwd*Fused`) | 2,874 | gfx942 2.46 |
| rocBLAS GEMM | 424 | gfx942 2.36, gfx90a 2.11 |
| Implicit GEMM, xdlops | 3,779 | gfx942 2.31, gfx90a 2.59 |
| Naive direct | 2,023 | gfx942 1.77, gfx90a 1.77 |

The `c` an output needs is the largest over every element of the output, and
every output up to 2^27 elements was analysed whole, so the maxima above are
exact for the data, not bounds (the bfloat16 summation coefficient the data
needs is at most 0.43). Re-measured at the current head on 5 seed sets on gfx942
and 3 on gfx90a: Winograd 5.68 and 4.95 (5.40 on gfx90a over 5 seed sets), CK
fused 2.46, rocBLAS 2.36 and 2.11, xdlops 2.31 and 2.59, naive 1.77 and 1.36.
See "Float against float64" for the same outputs graded against the exact sum.

### Float against the fixed 1e-5

The default that a float convolution-forward output had was `atol = rtol =
1e-5` (`conv::getToleranceFwd<float>()`; the 0.1 of the sections above is the
half and bfloat16 value). The new band is not simply tighter or looser: the
fixed band grows as `sqrt(K)` at a typical output (`1e-5 * (1 + |ref|)`, and
`|ref|` grows as `sqrt(K)`), the dynamic one as `K` (`sigma` and the summation
error both grow as `sqrt(K)`). For inputs uniform in [-1, 1] (`sigma = sqrt(K / 9)`),
the band at a typical output (`|ref| = sigma`) and the `atol` at an output near
zero, the fixed value first, then `c` = 4 (the base) and `c` = 9 (MIOpen's), with
the ratio to the fixed one in brackets:

| K | Typical output: fixed 1e-5 | c = 4 | c = 9 | Output near zero: fixed 1e-5 | c = 4 | c = 9 |
|---|---|---|---|---|---|---|
| 36 | 3.0e-5 | 6.2e-6 (0.21) | 1.3e-5 (0.45) | 1e-5 | 2.9e-6 (0.29) | 6.4e-6 (0.64) |
| 144 | 5.0e-5 | 2.4e-5 (0.48) | 5.3e-5 (1.05) | 1e-5 | 1.1e-5 (1.14) | 2.6e-5 (2.6) |
| 1,152 | 1.2e-4 | 1.9e-4 (1.5) | 4.2e-4 (3.4) | 1e-5 | 9.2e-5 (9.2) | 2.1e-4 (21) |
| 18,432 | 4.6e-4 | 2.9e-3 (6.4) | 6.6e-3 (14) | 1e-5 | 1.5e-3 (146) | 3.3e-3 (330) |

So the dynamic band is tighter than the fixed one at short reductions and
wider from `K` of about 540 (`c` = 4) or 130 (`c` = 9) at a typical output,
and from about 126 or 56 at an output near zero. It is wider on purpose:
long reductions need it. Of the correct float outputs in the calibration data
(a correct output has an error within 0.1, see above), the fixed 1e-5 fails
1,629 of the 7,746 distinct (case, seed, architecture) outputs, counted over
every element: none of the 1,432 outputs of develop's bundles, whose reductions
are at most 432 long, and 1,629 of the 6,314 of the test-plan bundles, with `K`
from 3 to 628,864: 1,283 of the 1,340 outputs with `K` above 512, and 346 of
the 4,974 with `K` of 512 or less.

What the wider band gives up is detection at long reductions: at `K` = 18,432
a float output that is wrong by less than 6.5e-5 of its size at a typical element
(1.5e-4 at `c` = 9) passes, where the fixed 1e-5 band caught a defect of 1.0e-5.
Float detection was measured on the dump data with the bugs of the half and
bfloat16 table, scored on the outputs whose bug-free result passes the
tolerance: a missing filter tap, a missing input channel, a shifted input and a
missing bias are caught in every output at 1e-5, at `c` = 9 (a missing tap:
10,312 to 10,316 of 10,316) and at `c` = 4 (non-Winograd outputs); one element
off by 4 ULP is caught by none (at most 4 of 9,133 at `c` = 4). Truncation was
not injected for float, and defects between 4 ULP and a missing tap were not
measured, so the band widths above remain the bound on what it can see. At a
typical output the base band is tighter than the fixed one up to `K` of about
540, and MIOpen's up to about 130.

### The base coefficient and the engine's

The base coefficient, which the SDK uses for an engine that sets nothing, is 4:
the smallest whole number that leaves a headroom of at least 1.5 (the floor used
for MIOpen's value below) over the largest need of any solver family but
Winograd, 2.59 on xdlops solvers on gfx90a (2.46 on CK fused solvers on
gfx942). At 4 that headroom is 1.54. 3 would still pass the data (1.16) but is
too thin for hardware nobody has measured. For Winograd the same constant fails
up to 5 correct outputs in a run
(one seed, one bundle set, one architecture); no theory term tells the two
apart, and the harness does not know the solver.

One coefficient scales both the `atol` and the `rtol` summation terms, and
that is a measured choice. Splitting them (the `rtol` term with a coefficient
`rho * c`, `rho` below 1) was fitted on the per-output need of the same data:
for `rho` = 0.5 the largest need of a non-Winograd solver is 2.67 and Winograd's
6.10, so the base stays 4 (a headroom of 1.50) and MIOpen's must go from 9 to
10, and the band at a typical output narrows 1.3 times at the base and 1.2
times at MIOpen's value. For `rho` = 0.25 the non-zero-mean cases (a fill that
is not centred on zero, where the partial sums follow `|ref|` and not
`sigma`) need 4.4, and for `rho` = 0 they need 23.7, so the base would have
to rise to 7 and 36. A split gains 1.3 times at most, and only by going below
the floor of 1.5 at the base; it gains 1.2 times at MIOpen's value; and with
less `rho` the cases that need `|ref|` break the floor. It is not worth a
second number. A term on `max(sigma, |ref|)` or on their root of squares was
not fitted: the data keeps the six worst elements of each output and not all of
them, and on those the `max` form needs at least 8.4 for Winograd against 5.7,
a gain of 1.4 at most at a typical element and less if another element needs
more. The term is kept as one coefficient on the sum of `sigma` and `|ref|`.

An engine whose solvers err more sets its own in its TOML, next to
`[[tolerance_overrides]]`:

```toml
[conv_fwd_tolerance.float]
summation_order = 9.0
```

`summation_order` is `c`. It must be a finite positive number. It replaces the
base coefficient for that engine's float outputs of that direction and changes
nothing for half and bfloat16. The other keys of the table, `lambda` and
`lambda_per_quartic_root_k`, the tables of the other directions and the rules
for the file are in "SDK defaults and an engine's allowance".

MIOpen's file sets 9. Its worst solver family, Winograd, needs 5.68 on gfx942
and 5.40 on gfx90a, so 9 leaves it a headroom of 1.58 and 1.67: at least 1.5,
a little above the 1.44 of half and bfloat16. The same 9 makes the other
families' bands wider than they need: their headroom is 3.6 or more.
The headrooms in this section are the minimum over every element of every
dumped output (see "Calibration").


### Which half solvers need lambda

Per output, the `lambda` it needs is the largest error left after `2 * eps *
|ref|` and `sqrt(K) * u32 * sigma`, over `eps * sigma`, over every element,
from the MIOpen log's solver of the output (half `ConvolutionFwd*` outputs of
develop's and the test-plan bundles, gfx942 and gfx90a, without rounding
points). Against the float64 result instead of the reference it is the same
within 0.1 (the reference needs none: at most 0).

| Solver | Outputs | `lambda` needed, max (p99) |
|---|---|---|
| `ConvWinoRageRxS<2-3>` | 235 | 8.09 (6.52) |
| `ConvWinoRageRxSFused<2-3>` | 257 | 8.05 (6.77) |
| `ConvBinWinogradRxSf2x3g1` | 321 | 6.80 (4.05) |
| `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC` | 1,025 | 5.21 (2.84) |
| `ConvBinWinogradRxSf2x3` | 61 | 4.75 (4.75) |
| `ConvBinWinogradRxSf3x2` | 40 | 3.56 (3.56) |
| `ConvHipImplicitGemm3DGroupFwdXdlops`, `ConvHipImplicitGemmGroupFwdXdlops`, `ConvCKIgemmGrpFwdActivFused`, `ConvDirectNaiveConvFwd`, `GemmFwdRest`, `GemmFwd1x1_0_1`, `GemmFwd1x1_0_2`, `ConvDepthwiseFwd2D` | 5,659 | none (at most 0.00) |

The solvers that keep intermediates in half are the five Winograd solvers and
`ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC`; the rest are within one rounding and
float accumulation (error beyond one rounding at most 0.27 `eps * sigma` rms).
The excess is not that of a half running sum, which grows as `sqrt(K)` and in
emulation needs `lambda` of 2.5 at `K` = 48 and more beyond: Winograd's falls
with `K` (8.09 up to 64, 5.25 up to 512, 4.22 up to 4,608, 1.21 above) and the
assembly solver's is 0 up to 64, 2.52, 2.63 and 5.21 for the ranges above. Both
are rms 0.5 to 0.9 `eps * sigma` beyond one rounding at every `K`, which is
what a fixed number of extra roundings of partial results (transformed tiles,
blocks) gives. Per solver and arch it is 8.09 on gfx942 and 6.80 on gfx90a for
Winograd. ALMIOPEN-3243 tracks the kernels. The same outputs by class of
kernel, the `lambda` needed at a headroom of 1:

| Class | gfx942 | gfx90a |
|---|---|---|
| Winograd | 8.09 | 6.80 |
| assembly implicit GEMM, split-K (`_gkgs`) | 5.21 (`K` = 18,432) | not logged (3.50 at most, all assembly) |
| assembly implicit GEMM, no split | 1.98 | - |
| CK, GEMM, naive, depthwise | none (at most 0) | none (at most 0) |

Winograd and the split-K assembly kernels are not conforming: they are MIOpen's
allowance, 12 for half (1.48 times Winograd's 8.09). The SDK's forward half
`lambda` is 3: 1.5 times 1.98, the largest need of a conforming solver, the
same factor that 12 has over Winograd's need. The gfx950 AOT conv-forward
engine (`hipkernel:Gfx950ConvFwdAot`), which sets no allowance, passes the same
346 cases at 3 as at 12 on an MI355X (gfx950), every conv-forward bundle case it
accepts.

A later check of that margin, on 5 new input seeds per architecture, MI308X
(gfx942) and MI250 (gfx90a), took the solver of each output from the MIOpen log
of the same run that produced it. The conforming solvers (`GemmFwdRest`,
`GemmFwd1x1_0_1`, the CK xdlops solvers, the non-split assembly implicit GEMM,
naive and depthwise) needed no `lambda` on either architecture, and at the SDK's
3 their smallest headroom was 2.59 on gfx942 and 2.88 on gfx90a, with no correct
output failing. Outputs without a logged solver were left out. A run of the
same seeds attributed through a separate find run looked tighter (1.05 and
1.18): that run's timing-based choice gave some outputs of Winograd and
split-K kernels to conforming solvers. Attribution for this fit needs the log
of the run whose outputs are graded.

### The cases the SDK bound failed

Twelve distinct cases (the 22 outputs of the first calibration, 32 at the
current head) exceed `calculateConvFpropTolerance<T, T, float>()` in the sense
of their largest error against its `atol`. The bound was given the true input
maxima in every one (the largest input of each is within 0.1 % of the sweep's
bounds). Causes, from the solver of each case and its elements:

- 7 bfloat16 `F21tiny` cases (inputs 1e-20: the products and the outputs, 1e-39,
  are below the smallest normal float): gfx90a with
  `ConvHipImplicitGemm3DGroupFwdXdlops`, `ConvHipImplicitGemmGroupFwdXdlops` and
  `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC`, which return 0 for the
  subnormal results (the reference, and float64, keep them: the error is the
  size of the output). On gfx942 the same case differs in 1 to 2 elements, one
  subnormal step of the reference's own sum. This is a flush of subnormals, not
  one of float accumulation, intermediates or the graph; the dynamic
  tolerance's `minNormal` term covers it.
- 3 half `F21tiny` cases (`s3` 1d and 2d, `s2` 1d: outputs at 6.4e-5 to 1.4e-4,
  next to the smallest normal half, 6.1e-5): `ConvBinWinogradRxSf2x3g1` on
  gfx90a, a Winograd solver, which does not sum the `K` products; it breaks the
  hypothesis of the bound at 729 to 13,471 elements.
- 2 half cases, `l17_1d` with a relu and `l17_2d`, `K` = 3 and 9, Winograd
  solvers on gfx942: they exceed the `atol` alone by up to 17 %, and pass when
  the number is also the `rtol`, as its callers use it (`|ref|` is 2.6 and 5.2).
  They are no failure of the bound.

No case is a rounding point of the graph and none is a bound given smaller
input maxima than the true ones. Under its hypothesis (float accumulation, one
rounding) the per-element bound `2 * gamma_K * sum|p_i| + eps_out * |y|` is
exceeded by 966 of the 16,232 half and bfloat16 outputs, 208 cases, all of the
same families: Winograd and the half solvers above, and the flushing ones.

### Float against float64

Graded against the float64 result instead of the sequential float32 reference,
with the exact value in the relative term, the `c` the engines need (maximum
and p99 over outputs, units `sqrt(K) * u32 * (sigma + |y|)`):

| Solver family | Against the reference | Against float64 |
|---|---|---|
| Winograd | 5.68 (4.97) | 5.76 (4.95) |
| CK fused | 2.46 (1.84) | 2.25 (1.62) |
| rocBLAS GEMM | 2.36 (2.07) | 1.85 (1.76) |
| Implicit GEMM, xdlops | 2.59 (2.07) | 2.30 (1.85) |
| Naive direct | 1.77 (1.14) | 1.98 (1.61) |

The reference itself needs up to 2.24 (p99 1.7) against float64, 2.05 up to
`K` of 64, 2.22 up to 512, 2.24 up to 4,608 and 1.57 above: a sequential float32
sum errs as much as the engines do. The `c` that the engines need does not fall
when the oracle is exact (at most 12 % for the non-Winograd families, and
Winograd's needs the same): their summation is a float sum of the same size,
not a better ordered one. So `c` = 4 and 9 keep their headroom (at least 1.54
and 1.58), and no detection comes back at long `K`: a missing filter tap, a
missing input channel and a shifted input are caught in every non-Winograd output
at `c` of 3 and above against either oracle (with `K` above 4,608: 97 of 98
missing taps), and one element off by 4 ULP in none at any `c` of 1 and above.

### Backward data and backward weights

Backward data (`ConvolutionBwd`, dgrad) and backward weights (`ConvolutionWrw`,
wrw) use the model of the sections above, the float form included (see "Float
outputs"): the same arithmetic model, the same two-step derivation of the
summation term (see "Where the summation term comes from"), the same fallback
for two constant operands, and the same statistical reading (see "What kind of
tolerance this is"). Three things differ: the two operands and `K` of their
own direction, their own `lambda` (the SDK's defaults and MIOpen's allowances
are in "SDK defaults and an engine's allowance"), and a float `lambda`. An
engine's `[conv_<direction>_tolerance.<type>]` tables apply to their own
direction only.

| Direction | Operands | `K` |
|---|---|---|
| forward | `x`, `w` | `C / groups` times the filter volume: every weight dimension but the first. |
| backward data | `dy`, `w` | The output channels of one group times, per spatial dimension, the most filter taps that reach one `dx` element: `ceil(R / (s / gcd(s, d)))` for filter extent `R`, stride `s` and dilation `d`. Padding and the border only remove taps. |
| backward weights | `x`, `dy` | The batch size times the spatial volume of `dy`. Products that read padding are zero, so for taps at the border this is an upper bound. |

`K` is an upper bound at the borders, as it is for the padded borders of
forward: backward data counts the most taps that reach one `dx` element and
backward weights counts the products that read padding, so an element that sums
fewer products gets a wider band than it needs. It is widest for a filter as
large as the image (the backward-weights cases `S13` and `S14`, a 16x16 and an
18x18 filter over a 16x16 image).

Derived and fitted. As for forward, the factor 2 on `eps * |ref|`, the form
`sqrt(K) * u32` of the summation term and `c` = 1 for half and bfloat16 are
derived; `c` = 4 for float is the forward base, a margin over the summation
theory (below). Every backward `lambda` is fitted. The SDK's is the smallest
value that leaves a headroom of at least 1.5 on the conforming solvers of the
calibration data, with `c` held at its value; MIOpen's is the smallest that
leaves 1.5 on every clean output, of both architectures (an arch table where
gfx90a needs more). Each traces to particular solvers (see "Which backward
solvers need lambda"), and none of them is a law of the arithmetic: a solver
that sums the `K` products in float and rounds once needs only the SDK's. The
backward-weights half and bfloat16 allowance of MIOpen grows as `K^(1/4)`
because the split-K kernels that set it split longer reductions into more
parts; the two coefficients are fitted together, to a headroom of 1.5 on the
data, with the smallest band at `K` = 1,152.

A backward activation fused behind a dgrad multiplies the gradient, the dgrad
result, by the activation's derivative at the forward input: at most 1 for relu,
sigmoid and tanh. Only `RELU_BWD`, `SIGMOID_BWD` and `TANH_BWD` are priced, the
backward modes that have a reference. The other five (`SOFTPLUS_BWD`,
`ELU_BWD`, `SWISH_BWD`, `GELU_BWD`, `GELU_APPROX_TANH_BWD`) leave the model and
the graph uses the per-op default. A priced one is modeled with slope 1, like
the forward activations, only when it reads the dgrad result as its gradient
(`in_0`) and not as its forward input, and only behind a `ConvolutionBwd`; its
forward input adds nothing to the size of its result. MIOpen runs no fused dgrad
graph, so no backward output with a rounding point or an activation was graded
in the calibration below.

At a typical output (`|ref|` = `sigma`, inputs uniform in [-1, 1], `K` =
1,152) the band, divided by `sigma`, is, for each way of setting the tolerance
(the per-op default and the worst-case bound used as both `atol` and `rtol`;
the dynamic tolerance at the SDK's defaults and at MIOpen's allowance):

| Direction | Output | per-op default | worst-case bound | dynamic, SDK defaults | dynamic, MIOpen |
|---|---|---|---|---|---|
| backward data | float | 9.25e-6 | 0.344 | 1.8e-5 | 1.82e-5 |
| backward data | half | 0.022 | 1.6 | 0.0078 | 0.015 |
| backward data | bfloat16 | 0.022 | 10.1 | 0.039 | 0.063 |
| backward weights | float | 2.18e-4 | 0.344 | 1.7e-5 | 1.82e-5 |
| backward weights | half | 0.218 | 1.6 | 0.013 | 0.029 |
| backward weights | bfloat16 | 0.218 | 10.1 | 0.086 | 0.11 |

The backward-weights half band is, at MIOpen's allowance, 0.020 of `sigma` at
`K` = 16, 0.054 at 65,536 and 0.107 at 2 million (the SDK's: 0.013 at every
`K`); bfloat16 is 0.055, 0.27 and 0.61 (the SDK's: 0.086).

Where the dynamic band is wider than the per-op default (float backward data,
bfloat16 backward data), the default fails correct output; see the
calibration below.

### Calibration: backward data and backward weights

Data: the float, half and bfloat16 cases MIOpen runs of the conv-dgrad
test-plan bundles (`ConvolutionBwd`, `ConvolutionBwdPointwise`; no fused case
executes), of the same problems posed as backward weights (x the shape of dx,
dw the shape of w; calibration bundles not in the tree) together with the
backward-weights bundles in the tree, at all four levels. gfx942 (MI325X and
MI308X): 6 input seeds. gfx90a (MI250): 4 input seeds, and 3 more for the
outputs above 4 M elements. Each output was compared with the harness reference
and with a float64 recomputation from the same inputs, over every element.
Outputs above 4 M elements are analysed whole in both directions (backward
weights up to `K` of 2.8 million), so their headroom is exact. The backward-data
ones on gfx942, 6 seeds, at MIOpen's allowance:

| Output | Outputs with a tensor above 4 M elements | `K` | Fail | Smallest headroom |
|---|---|---|---|---|
| half | 630 | 2 to 628,864 | 0 | 1.54 |
| float | 348 | 2 to 628,864 | 0 | 1.54 |
| bfloat16 | 336 | 3 to 628,864 | 0 | 3.31 |

Not in the data: the mixed-type class `F20` (MIOpen faults on it on gfx942:
ALMIOPEN-3302) and the constant-fill class `F21const` (both operands 1.0), which
takes the per-op default.

Outputs MIOpen computes wrongly are not data for the fit: an output whose
error against the float64 result exceeds a quarter of `sigma`, or that holds a
NaN or an infinity. That leaves out the mixed-layout class `F15`
(ALMIOPEN-3298), the non-packed-stride classes `F16plane`, `F16chan` and
`F16fast` (ALMIOPEN-3300), and the extreme-magnitude class `F21` (`F21tiny` and
`F21large`: values that underflow or overflow), the last on both architectures.
On gfx90a, 13 float32 backward-weights cases (the multipass Winograd
backward-weights solver and 4 assembly implicit-GEMM shapes) err by about 3e-5
to 1.9e-4 relative, and two bfloat16 assembly backward-weights outputs were
wrong on one seed: MIOpen accuracy defects, not data for the fit. These cases
and the failing half `F21` backward cases are in the conv-dgrad test-plan
bundles, not in this tree, and their gfx90a skips go with those bundles. The
left-out tests number 455 on gfx942 and 525 on gfx90a.

| Arch | Direction | Output | Correct outputs | Fail the per-op default | Fail MIOpen's allowance | Smallest headroom (MIOpen's) | Conforming outputs | Fail the SDK defaults | Smallest headroom (SDK, conforming) |
|---|---|---|---|---|---|---|---|---|---|
| gfx942 | backward data | float | 3,758 | 1,004 | 0 | 1.54 | 2,944 | 0 | 1.55 |
| gfx942 | backward data | half | 4,265 | 244 | 0 | 1.54 | 2,213 | 0 | 1.61 |
| gfx942 | backward data | bfloat16 | 3,566 | 278 | 0 | 1.56 | 3,033 | 0 | 1.78 |
| gfx942 | backward weights | float | 4,008 | 447 | 0 | 1.55 | 3,659 | 0 | 1.64 |
| gfx942 | backward weights | half | 4,525 | 155 | 0 | 1.55 | 1,177 | 0 | 1.55 |
| gfx942 | backward weights | bfloat16 | 3,831 | 425 | 0 | 1.55 | 2,409 | 0 | 1.60 |
| gfx90a | backward data | float | 2,930 | 756 | 0 | 1.52 | 2,430 | 0 | 2.16 |
| gfx90a | backward data | half | 3,239 | 1,343 | 0 | 1.50 | 1,344 | 16 | 0.54 |
| gfx90a | backward data | bfloat16 | 2,776 | 202 | 0 | 3.22 | 2,686 | 0 | 2.28 |
| gfx90a | backward weights | float | 3,052 | 237 | 0 | 1.55 | 2,773 | 0 | 2.08 |
| gfx90a | backward weights | half | 3,431 | 515 | 0 | 1.51 | 801 | 8 | 0.78 |
| gfx90a | backward weights | bfloat16 | 2,945 | 3 | 0 | 3.08 | 2,489 | 0 | 4.02 |

"Conforming" outputs are those of solvers that sum the `K` products in float
and round once, from the MIOpen log's solver of each output; Winograd, the
split-K kernels that add partial results in the output type, the gfx90a fp16
alternate implementation and outputs with no attributed solver are not counted
(see "Which backward solvers need lambda"). The SDK defaults fail none of them
except 24 gfx90a rows, 16 of half backward data and 8 of half backward weights,
all attributed to the naive solvers, with errors at bfloat16 level. That is the
error of the alternate implementation, not of a naive kernel: the solver log
comes from a separate find run whose choice of kernel is timing-based, and the
runs that produced these outputs used the alternate-implementation kernels, so
the attribution of these rows is the uncertain part.

Headroom here is exact for every analysed output: the smallest factor by which
the whole band `atol + rtol * |ref_i|` could shrink, over every element of the
output, before the output fails. As for forward, a fixed constant is a fixed
number of standard deviations per element, so the chance that a correct output
fails grows with its element count.

### Which backward solvers need lambda

`lambda` needed, in units of `eps * rms`, is the error left after `2 * eps *
|ref|` and the summation term, over `eps * rms`, over every element, from
the MIOpen log's solver of each output (the calibration set above; outputs with
no attributed solver are in the allowance fit and not in the SDK's). The largest
need per class of solver and the allowances:

| Direction, output | Not conforming: largest need, gfx942 / gfx90a | Conforming: largest need, gfx942 / gfx90a | SDK | MIOpen |
|---|---|---|---|---|
| backward data, half | Winograd (`ConvWinoRageRxS<2-3>`, `ConvBinWinogradRxSf2x3`, `Sf2x3g1`, `Sf3x2`) 8.34 / none; CK split-K (`ConvHipImplicitGemmGroupBwdXdlops`) 3.05 / 12.01; assembly split-K fp16 2.81 / none; fp16 alternate implementation (`GemmBwdRest`, `GemmBwd1x1_stride1`, `GemmBwd1x1_stride2`, `ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC`) none / 27.87 | 3.68 (assembly implicit GEMM, `K` = 128) / 15.98 (naive direct, with the attribution caveat above) | 6 | 13 (gfx90a 46) |
| backward data, bfloat16 | CK split-K (`ConvHipImplicitGemmGroupBwdXdlops`) 2.33 / none | 1.65 (assembly implicit GEMM, `K` = 2,304) / 1.25 (`GemmBwdRest`) | 3 | 6 |
| backward weights, half | assembly split-K fp16 (`ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC`) 43.6 at `K` = 1.6 million (36.6 at 2.8 million, split not logged) / none; fp16 alternate implementation (the assembly solver again, `GemmWrw1x1_stride1`, `GemmWrwUniversal`) none / 38.1 at 1.6 million; CK split-K (`ConvHipImplicitGemm3DGroupWrwXdlops`) 14.0 / 17.2; Winograd 9.48 / none | 5.64 (`GemmWrwUniversal`, `K` = 13,952) / 16.15 (naive direct, with the attribution caveat above) | 11 | 13 + 2.5 `K^(1/4)` (gfx90a 35 + 2 `K^(1/4)`) |
| backward weights, bfloat16 | CK split-K (`ConvHipImplicitGemm3DGroupWrwXdlops`, `ConvHipImplicitGemmGroupWrwXdlops`) 14.1 at `K` = 524,288 / none | 4.92 (`GemmWrw1x1_stride1`, `K` = 32,768) / 1.08 | 9 | 1 + 2 `K^(1/4)` |
| backward data, float | Winograd (`ConvBinWinogradRxSf2x3g1`, `Sf3x2`) at `K` = 2 to 12: 6.52 / 5.86 | 7.14 (`GemmBwd1x1_stride1`, `K` = 2) / none needed | 13 | 15 (gfx90a 18) |
| backward weights, float | Winograd (`ConvBinWinogradRxSf2x3`, `Sf2x3g1`) at `K` = 1 to 2: 8.76 / 8.76 | 1.28 (assembly implicit GEMM, `K` = 4) / none needed | 5 | 15 |

A value in the allowance columns is not 1.5 times the need: the headroom is
that of the whole band, `rtol` included, at the element where it is smallest.

The backward-weights half and bfloat16 need is not that of a half running
sum, which grows as `sqrt(K)`. It comes from split-K kernels that add their
partial results in the output type in a non-deterministic order: two
identical runs differ in 62 to 73 % of the elements, the error's rms equals
that of an io-type running sum over `G` = 8 to 128 partials (the split count
the MIOpen log shows) and not over `K`. Longer reductions get more partials,
so the need grows with `K`: its largest value per range of `K` rises about as
`K^(1/4)`, from 9 below `K` = 4,096 to 22 near 8,192, 26 near 1 million and
43 at 1.6 million, where one correct output of a single run, of 7,168
elements, fell outside a flat `lambda` of 36. The same
kernels on an fp32 workspace (the assembly solver for bfloat16, and for half
when its split is not packed) need about 0. This is a kernel that does not
honour the float compute type; ALMIOPEN-3380 tracks it, and MIOpen's growing
`lambda` is what the tolerance pays for it until it is fixed. Its size
depends on the GPU: the split count follows the CU count, and the largest
needs are from an MI325X. Beyond `K` of 2.8 million the growth is an
extrapolation.

On gfx90a the half kernels have a second class. MIOpen's backward fp16
alternate implementation, which the GEMM and xdlops implicit-GEMM solvers use
(`GemmBwdRest`, `GemmBwd1x1_*`, `ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC`,
`ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC`, `GemmWrw1x1_stride1`,
`GemmWrwUniversal`), rounds the operands to bfloat16 to avoid the MI200 MFMA
fp16 denormal flush. Its error is about 6 bfloat16 roundoffs (2^-8) of `sigma`,
where the CK, naive and depthwise solvers, which stay at half precision, leave
about 0.4. It is not an io-type sum either, so it is MIOpen's allowance and
not the SDK's: the gfx90a tables of backward-data half (46) and
backward-weights half (35 + 2 `K^(1/4)`).

The float `lambda` covers what the summation term leaves at the shortest
reductions, where 2 or 3 products of a correct output differ from the reference
by up to 10 `sqrt(K) * u32 * sigma`. Winograd is the class MIOpen's allowance
(15; 18 for backward data on gfx90a) is for: a Winograd solver does not sum the
`K` products. The SDK's covers the conforming solvers: their largest need is
the 1x1 GEMM's at `K` = 2 on gfx942 for backward data (7.1; the SDK's is 13) and
the assembly implicit GEMM's for backward weights (1.3; the SDK's is 5). The
other conforming families need a negative `lambda` at `c` = 4. Covering
Winograd with `c` alone would take `c` of about 12
(backward data) and 22 (backward weights), three to five times wider at large
`K`.

### Backward summation coefficient against the theory and float64

For `K` above 64, where the summation term carries the float band, the `c`
needed (single-constant form of "Float against float64", against the
reference) is at most 2.96 for backward data and 1.98 for backward weights,
rising with the output's element count as the derivation says (median 0.6 to
1.3 and 0.6 to 1.4 from 1e3 to 4e6 elements). The derivation's figure for
two orderings of ideal float arithmetic at those counts is 1.8 to 2.9, so `c`
= 4 is a margin over theory here too. Per solver family, against the reference
and against the float64 result:

| Direction | Solver family | Against the reference | Against float64 |
|---|---|---|---|
| backward data | CK implicit GEMM | 1.95 | 1.93 |
| backward data | assembly implicit GEMM | 2.07 | 1.67 |
| backward data | rocBLAS GEMM | 2.15 | 1.86 |
| backward data | naive direct | 1.66 | 1.86 |
| backward data | Winograd | 4.59 | 3.91 |
| backward weights | CK implicit GEMM | 1.57 | 1.36 |
| backward weights | assembly implicit GEMM | 1.93 | 1.70 |
| backward weights | rocBLAS GEMM | 1.70 | 1.63 |
| backward weights | naive direct | 1.26 | 1.20 |
| backward weights | Winograd | 10.10 | 10.13 |

(One seed, every element; the reference itself needs up to 1.90 against
float64.) As for forward, the exact oracle lowers the need by at most about
0.5 for the non-Winograd families and not at all for Winograd, whose error is
its own: `c` stays 4.

Two operands with a nonzero mean (uniform on `[0, a]`: the class `F21large`,
the only such class in the data of the earlier gfx942 fit, now excluded) are
where `rms` matters. Float and backward data do not need it: without it the
`|ref|` term of the band carries the mean, as the derivation says (`c` needed
0.75 to 1.0). The backward-weights half and bfloat16 outputs do: their split-K
error grows with the partial sums, which with a nonzero mean grow with `|ref|`,
and without a magnitude that includes the mean they would need `lambda` of 64 or
an `rtol` of 5 `eps`. That is why an engine's `lambda` is priced on `rms`.

### The cases the SDK bound failed, backward

Five half cases (16 outputs over the seeds) exceed the worst-case bound
(`calculateConvDgradTolerance<T, T, float>()` and
`calculateConvWrwTolerance<T, T, float>()`, as `atol` and `rtol`), all at `K` =
2 or 3: backward data `L37`, `L77` and `R125` (one seed, on the GPU where
MIOpen picks Winograd for it) and backward weights `S13` and `S14` (a 16x16 and
an 18x18 filter over a 16x16 image). Every one runs a Winograd solver
(`ConvWinoRageRxS<2-3>`, `ConvBinWinogradRxSf2x3g1`, `ConvBinWinogradRxSf2x3`):
not a sum of the `K` products, with half transform intermediates, an error of
about 0.7 `eps * sigma` rms whatever `K` is, against a bound of 2 to 3 `eps`.
The bound had the true input maxima, and none involves a graph rounding or a
subnormal flush. No bfloat16 or float output exceeds it.

Detection, measured by injecting each bug into the exact result and grading
it against the real reference (gfx942, the fit seeds; the fraction of mutated
outputs that fail). gfx90a is within 4 points of every cell. Where the per-op
default is narrower (half and bfloat16 backward data) it catches more io-type
accumulators, at the price of failing correct outputs (table above):

| Direction, output | Injected bug | per-op default | MIOpen's allowance | SDK defaults |
|---|---|---|---|---|
| backward data, float | one filter tap / one output channel missing | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % |
| backward data, float | inputs rounded to tf32 / to bfloat16 | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % |
| backward data, float | one element off by 4 ULP / output truncated | 27 % / 27 % | 0 % / 0 % | 0 % / 0 % |
| backward data, half | one filter tap / one output channel missing | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % |
| backward data, half | accumulator in the io type | 24 % | 3 % | 17 % |
| backward data, half | one element off by 4 ULP / output truncated | 0 % / 0 % | 0 % / 0 % | 28 % / 0 % |
| backward data, bfloat16 | one filter tap / one output channel missing | 100 % / 100 % | 100 % / 99 % | 100 % / 99 % |
| backward data, bfloat16 | accumulator in the io type | 68 % | 16 % | 35 % |
| backward data, bfloat16 | one element off by 4 ULP / output truncated | 54 % / 0 % | 25 % / 0 % | 55 % / 0 % |
| backward weights, float | one batch sample / one dy row missing | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % |
| backward weights, float | inputs rounded to tf32 / to bfloat16 | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % |
| backward weights, float | one element off by 4 ULP / output truncated | 10 % / 10 % | 0 % / 0 % | 0 % / 0 % |
| backward weights, half | one batch sample / one dy row missing | 100 % / 99 % | 100 % / 100 % | 100 % / 100 % |
| backward weights, half | accumulator in the io type | 1 % | 0 % | 0 % |
| backward weights, half | one element off by 4 ULP / output truncated | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| backward weights, bfloat16 | one batch sample / one dy row missing | 100 % / 99 % | 100 % / 94 % | 100 % / 99 % |
| backward weights, bfloat16 | accumulator in the io type | 48 % | 0 % | 1 % |
| backward weights, bfloat16 | one element off by 4 ULP / output truncated | 0 % / 0 % | 2 % / 0 % | 1 % / 0 % |

An accumulator in the io type is what the allowance of half and of bfloat16
backward weights allows for, so it is not detected there: that is the cost of
the split-K kernels of ALMIOPEN-3380, and the SDK's defaults do not detect it
either. With a bfloat16 backward-weights `lambda` of about 13 at `K` = 1,152
(MIOpen's 1 + 2 `K^(1/4)`; the SDK's is 9), one missing product of a 1,152-product
sum is not detected either; a missing batch sample or row of `dy` still is.

### Harness runs, backward

The bundle harness with the MIOpen engine at the constants of this document (the
SDK's defaults and MIOpen's file), every level, the conv-dgrad test-plan bundles
and the backward-weights bundles derived from them together with every conv
bundle in the tree, two input seeds (the bundles' own and one not used in the
calibration), the classes `X3` and `F20` left out:

| GPU | Direction | Cases run (2 seeds) | Passed | Failed |
|---|---|---|---|---|
| MI300-series (gfx942) | backward data | 5,088 | 4,915 | 173 |
| MI300-series (gfx942) | backward weights | 5,366 | 5,211 | 155 |
| MI250 (gfx90a) | backward data | 5,088 | 4,937 | 151 |
| MI250 (gfx90a) | backward weights | 5,366 | 5,176 | 190 |

Every backward failure is an output MIOpen computes wrongly, the same cases on
both seeds: the classes `F15`, `F16plane`, `F16chan` and `F16fast` (values of
1e4 to 1e38, infinities, NaNs, wrong signs or doubled values), `F21large` and
`F21tiny` outputs that overflow or underflow, and on gfx90a the float
backward-weights cases of the multipass Winograd and asm igemm solvers and
bfloat16 backward-weights outputs of the asm igemm solver on NCHW 3x3 shapes
that are off by 0.7 to 27 on outputs of 2 to 53 (see "Calibration: backward
data and backward weights"). No correct output fails. The forward outputs in the
same runs are graded as before this change: the failures there are graphs the
engine does not execute and one 3D float case graded by a `[[tolerance_overrides]]`
entry of MIOpen's file, not by the dynamic tolerance.

### When an engine needs more

The SDK's allowance describes kernels that sum the `K` products in float and
round once. An engine whose kernels do not (Winograd, split-K kernels that add
partial results in the output type, operands rounded below the io type) sets
its own allowance in its TOML (see "SDK defaults and an engine's allowance"):
find the solver and the element that failed, measure the `lambda` it needs as in
"Which backward solvers need lambda", and name the ticket that tracks the
accuracy in the comment next to the value, as MIOpen's file does. A
`[[tolerance_overrides]]` entry is for what no allowance covers: a kernel that
errs more than the model describes, or that is wrong. Find the solver and the
element that failed before you add one, and name the ticket that tracks the
accuracy in its comment, as for any override.
