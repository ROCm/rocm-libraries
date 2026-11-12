# Tolerances: how an engine's output is graded

Every output tensor of a bundle is compared element by element with its oracle
(golden data or a reference executor):

```text
|impl_i - ref_i| <= atol + rtol * |ref_i|      for every element i
```

One element outside the band fails the tensor. This document says where
`atol` and `rtol` come from, and explains the dynamic tolerance that grades
half, bfloat16 and float convolution-forward outputs.

## Where the numbers come from

For each output tensor, the bundle harness takes the first of these that
applies:

| Source | Applies to | Where |
|---|---|---|
| A `[[validator_overrides]]` entry | The tensors its glob names | The engine's TOML (see [File Formats](file-formats.md)) |
| A `[[tolerance_overrides]]` entry | The tests its glob names | The engine's TOML |
| The convolution-forward dynamic tolerance | half, bfloat16 and float outputs of a graph with one `ConvolutionFwd` node that accumulates in float, whose pointwise nodes and intermediates are half, bfloat16, float or double | `src/harness/tolerance/ConvFwdDynamicTolerance.hpp` |
| The per-op default | Everything else; the loosest node of the graph wins, and the same value is used for `atol` and `rtol` | `TestTolerances.hpp` through `src/harness/tolerance/ToleranceResolver.hpp` |

TOML overrides describe how far one engine may drift. They are never applied
when golden data is checked against a reference executor, and neither is the
dynamic tolerance: only the engine's output is graded with it, and golden data
is checked against a reference executor with the per-op default.

## The convolution-forward dynamic tolerance

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

The worst-case bound in the test SDK, `calculateConvFpropTolerance()`, scales
with the problem, but assumes an output can be as large as
`K * max|x| * max|w|`. Its callers use the one number as both `atol` and
`rtol`. With inputs of both signs a real output grows only as `sqrt(K)`, so
the band the bound makes, which grows as `K` relative to a typical output, is
wider than that output from `K` of about 100 on for bfloat16 (about 800 for
half), and passes almost anything.

For float the per-op default is 1e-5 for both, not 0.1, and the comparison
goes the other way at long reductions: see "Float against the fixed 1e-5".

### The model

For an output stored as half or bfloat16, computed with float accumulation,
and graded against a float-accumulating reference (float adds a summation term
to `rtol` and takes a coefficient: see "Float outputs"):

```text
|impl_i - ref_i| <= 2 * eps * |ref_i|                      rtol
                  + (sqrt(K) * u32 + lambda * eps) * sigma  atol
                  + sum over rounding points of s * eps_p * m
                  + minNormal
```

| Symbol | Meaning |
|---|---|
| `K` | Products summed per output: the weight tensor's channel dimension (C / groups) times the filter volume. |
| `sigma` | The typical output magnitude, `sqrt(K * mean(x^2) * mean(w^2))`, times the product, over the fused multiplies, of the largest magnitude of the tensor each one applies. The mean squares come from how the inputs were filled (see below). |
| `eps` | Machine epsilon of the output type: 2^-7 for bfloat16, 2^-10 for half. |
| `u32` | Unit roundoff of float, 2^-24. |
| `lambda` | 0 for bfloat16, 12 for half (calibrated, see below). |
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
  where the relative term is small.
- **Intermediates in the io type** (`lambda * eps * sigma`). Some half kernels
  keep partial results in half (for example Winograd transforms). bfloat16
  showed none.
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

`sigma` grows as `sqrt(K)` exactly as the output does, so the band is a fixed
fraction of the output at every reduction length. At a typical output
(`|ref|` = `sigma`) the band, divided by `sigma`, is `atol / sigma + rtol`.
For inputs uniform in [-1, 1] it is, for each way of setting the tolerance
(the worst-case bound is used as both `atol` and `rtol`, as its callers do):

| Output | K | fixed 0.1 | worst-case bound | dynamic |
|---|---|---|---|---|
| bfloat16 | 16 | 0.175 | 0.22 | 0.016 |
| bfloat16 | 1,152 | 0.109 | 10.1 | 0.016 |
| bfloat16 | 65,536 | 0.101 | 535 | 0.016 |
| half | 16 | 0.175 | 0.027 | 0.014 |
| half | 1,152 | 0.109 | 1.6 | 0.014 |
| half | 65,536 | 0.101 | 82 | 0.014 |

The worst-case column prices `calculateConvFpropTolerance<T, T, float>()`, as
the sample passes it. With `double` compute, as the GPU reference tests pass
it, bfloat16 is 9.8 and 518 and half 1.2 and 65 at the two larger `K`.

The worst-case bound is tighter than the dynamic tolerance only for half at
`K` of 6 or less. In the calibration data it also failed 22 of the 10,434
correct half and bfloat16 outputs graded (12 distinct cases): 2 of the 5,091
of the fit seeds and 2 of the 3,394 of the held-out seeds on gfx942 (one half
and one bfloat16 each), and 18 of the 1,949 on gfx90a.

The function is `calculateConvFpropOutputTolerance<T>()` in
`projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp`.

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
in the last place, `eps_p / 2 * |v|` at most. The elements of a tensor reach
about twice their root mean square `m`, so one point contributes `eps_p * m`
at its own place, and the error reaches the output multiplied by the slope
`s` of the nodes in between: 1 through `ADD`, `SUBTRACT` and the modes assumed
to have a slope of about 1, and the largest magnitude of the scale through
a `MUL`. The error in the scale operand of a `MUL` that is itself rounded
reaches the output multiplied by the size `m` of the convolution's side of
the multiply. Where several nodes read a tensor and their results go to
different outputs, the slope is the largest of them. Where the results meet
again, one error travels both paths and the contributions add: for
`ADD(MUL(y, s), y)` with `max|s|` = 3 the slope of `y` is 3 + 1 = 4.
The points add (they do not combine as a root of squares: with five points
the sum is at most 2.2 times larger, and the data does not tell the two apart):

```text
atol += sum over the points of  s * eps_p * m
```

The magnitude `m` follows the graph from the convolution's `sigma` (without
the scales). A value is a factor times the convolution's result plus a
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
The coefficient 1 comes from that bound and is not fitted: the data needs at
most 0.115 of the term (see "Calibration of the rounding points").

### What the harness feeds it

`convFwdErrorScale()` applies the model only when the graph has exactly one
`ConvolutionFwd` node, with float compute type, and its `x` and `w` are graph
inputs: not virtual, and not written by another node. Every pointwise node
must compute in half, bfloat16, float or double, and every intermediate must
be stored as one of these: a tensor that one node writes and another reads
(the convolution's result, the pointwise results), virtual or not; an unset
type is none of them. A result that no node reads is a graph output and is
rounded once, as the model assumes, unless a node computing below float
stored it in another type (see "Rounding points"). A value stored or computed in half or
bfloat16 is a rounding point (see "Rounding points"); float and double add
none. A `MUL` node with one
operand from the convolution and the other a graph input (in either operand
order) scales `sigma` by that input's largest magnitude, and the scales of
several multiplies multiply; a `MUL` of anything else (the result with itself,
two inputs) leaves the model. The other pointwise modes are not examined: each
is assumed to change an error by at most its slope, about 1, so it leaves
`sigma` unchanged. The conv-fwd test-plan bundles the calibration used have
`ADD` and `MUL`, and as activations
`RELU_FWD` (plain, with an upper or a lower clip, or leaky), `IDENTITY`,
`SIGMOID_FWD`, `TANH_FWD`, `SOFTPLUS_FWD`, `ELU_FWD`, `SWISH_FWD`, `GELU_FWD`
and `GELU_APPROX_TANH_FWD`: 506 of their 1,132 half and bfloat16
`ConvolutionFwdPointwise` cases use an activation other than `RELU_FWD`. At the
parameters those bundles set (softplus and swish beta 1, elu alpha 1) the largest
slope among them is about 1.13 (gelu). A mode with another gain, such as `DIV`
or `RECIPROCAL`, would be graded as if it had none. Nodes that are neither the
convolution nor pointwise are ignored. The scale is one value for the whole
graph, not one per output; those graphs have at most one multiply.
A graph outside what is described here, an input whose statistics are not
available (a NaN or an infinity in golden data or in a fill recipe, an
unreadable baked value) use the per-op default.

A graph input's statistics come from, in this order:

- the value baked into the graph, for a scalar constant (in the test-plan
  bundles 40 of the 48 Variant3 cases, 27 of the 33 in half and bfloat16, bake
  their scale as 0.5; the `F27runtime` and `F27channel` cases fill it
  instead): `v^2` and `|v|`;
- the fill recipe, for a tensor the harness fills: `(lo^2 + lo*hi + hi^2) / 3`
  for a uniform draw on `[lo, hi]`, `v^2` for a fixed value, the mean square
  of the list for a value list, so no tensor is read;
- a scan on the host, for an input loaded from golden blobs, which is
  already there.

`sigma` is the typical output only for inputs with a zero mean. Then partial
sums cancel and the output grows as `sqrt(K)`. For a nonzero mean (a constant
fill, or uniform on `[0, a]`) the output is larger than `sigma` (the root mean
square is 3, 25 and 190 times `sigma` for uniform on `[0, 1]` at `K` = 16,
1,152 and 65,536), so `atol` is a smaller fraction of it and the `rtol` term,
which scales with the output, carries the tolerance. The calibration covered
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
  noise, at most 0.4 `u32 * sqrt(K) * sigma`, flat in `K`. The largest error
  of any bfloat16 output is within the relative term: one rounding step.
- half: MIOpen half kernels leave up to `8 * eps * sigma` on outputs near zero,
  mostly NCHW cases with short reductions. `lambda = 12` is 1.5 times the
  largest value seen.

Validation on data the fit did not see:

| Data | Outputs graded | Fail the dynamic tolerance | Smallest headroom |
|---|---|---|---|
| gfx942, 2 new input seeds | 3,394 | 0 | 1.44 |
| gfx90a (MI250), 2 seeds | 1,949 | 0 | 2.0 |

Headroom is the tolerance divided by the error, at the element of an output
where that ratio is smallest: the factor by which the error could grow before
the output fails. It is 1 at the point of failure. It was taken over the 48
largest-error elements (by absolute error and by rounding steps) and 48
random elements of each output; the failure counts use every element.

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

How much tighter it is: the median headroom per output, by range of `K`, is
2.0 to 2.7 for bfloat16 and 3 to 12 for half. For the fixed 0.1 it is 15 to
66 (bfloat16) and 4 to 160 (half), and for the older bound it passes 5,000 at
large `K`.

Against the fixed 0.1 per element (a band of 0.1 + 0.1 * `|ref|`), bfloat16 is
tighter at every element and every `K`: for inputs uniform in [-1, 1] its
`atol` is 0.0013 at `K` = 65,536 and its `rtol` is 0.0156. half is tighter
relative to a typical output at every `K` (1.4 % of it, against 10 % to
17.5 %), but not for an element near zero, where the band is the `atol`. The
`atol` grows with `sigma`, which is `sqrt(K / 9)` for those inputs, and passes
0.1 from `K` of about 650 on: 0.133 at `K` = 1,152 and 0.53 at 18,432. At
`K` = 18,432 (`sigma` about 45) the dynamic band is wider than the fixed one
for outputs smaller than about 4.4 in magnitude. That is the intended
behavior and it cannot be capped at 0.1: the error of the half kernels in
absolute terms grows with the size of the outputs, and the correct gfx90a
output above, 0.12 off at `K` = 18,432, is already outside the fixed 0.1.

Detection, measured by injecting each bug into the exact result and grading
it against the real reference (gfx942, 3 seeds):

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
inputs uniform in [-1, 1], and is about 0.5 at `K` = 16,000. A missing filter
tap is `sqrt(C)` products against an output of `sqrt(C * R * S)` of them, so
how much of the half `atol` it fills depends on the filter volume `R * S`, not
on `K` alone: its ratio to the `lambda` term is `1 / (12 * eps * sqrt(R * S))`
for uniform fills, above 1 only while `R * S` is below about 7,000 (computed,
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
conditions as half and bfloat16. The output has no `lambda` term, and the order
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
| rocBLAS GEMM | 424 | gfx942 2.35, gfx90a 2.11 |
| Implicit GEMM, xdlops | 3,779 | gfx942 2.31, gfx90a 2.31 |
| Naive direct | 2,023 | gfx942 1.77, gfx90a 1.77 |

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
1,561 of the 10,451 distinct (case, seed, architecture) outputs, counted from the
largest errors and the sampled elements of each output, so at least that many:
none of the 2,237 outputs of develop's bundles, whose reductions are at most 512
long, and 1,561 of the 8,214 of the test-plan bundles, with `K` from 128 to 93,312:
1,470 of the 1,704 outputs with `K` above 512, and 91 of the 6,510 with `K`
of 512 or less.

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
Winograd, 2.46 on CK fused solvers on gfx942. At 4 that headroom is 1.63. 3
would be enough to pass the data (1.22) but is too thin for hardware nobody has
measured. For Winograd the same constant fails up to 5 correct outputs in a run
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

`summation_order` is `c`. It must be a finite positive number: a file with
another value, with another output type (the table is per output type, so that
the half `lambda` can take the same shape later) or with another key does not
load, and neither does one with a top-level table other than `[meta]`,
`[[tolerance_overrides]]`, `[[validator_overrides]]`, `[[test_skips]]` and
`[conv_fwd_tolerance]` (a misspelt section name is an error, not a missing
setting). It replaces the base coefficient for that engine's float outputs and
changes nothing for half and bfloat16. A `[[tolerance_overrides]]` entry that
matches a test still replaces the dynamic tolerance whole, as for half and
bfloat16. Only the bundle harness reads the table; the C++ graph harness does
not grade with the dynamic tolerance.

MIOpen's file sets 9. Its worst solver family, Winograd, needs 5.68 on gfx942
and 5.40 on gfx90a, so 9 leaves it a headroom of 1.58 and 1.67: at least 1.5,
a little above the 1.44 of half and bfloat16. The same 9 makes the other
families' bands wider than they need: their headroom is 3.6 or more.


### When an engine needs more

The model describes float accumulation and, for half, io-precision
intermediates of the size MIOpen shows. An engine whose kernels are less
accurate fails here first. Find the solver and the element that failed
before you add a `[[tolerance_overrides]]` entry for those cases, and
name the ticket that tracks the accuracy in its comment, as for any override.
