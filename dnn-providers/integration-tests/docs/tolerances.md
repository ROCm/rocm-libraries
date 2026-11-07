# Tolerances: how an engine's output is graded

Every output tensor of a bundle is compared element by element with its oracle
(golden data or a reference executor):

```text
|impl_i - ref_i| <= atol + rtol * |ref_i|      for every element i
```

One element outside the band fails the tensor. This document says where
`atol` and `rtol` come from, and explains the dynamic tolerance that grades
half and bfloat16 convolution-forward outputs.

## Where the numbers come from

For each output tensor, the bundle harness takes the first of these that
applies:

| Source | Applies to | Where |
|---|---|---|
| A `[[validator_overrides]]` entry | The tensors its glob names | The engine's TOML (see [File Formats](file-formats.md)) |
| A `[[tolerance_overrides]]` entry | The tests its glob names | The engine's TOML |
| The convolution-forward dynamic tolerance | half and bfloat16 outputs of a graph with one `ConvolutionFwd` node that accumulates in float | `src/harness/tolerance/ConvFwdDynamicTolerance.hpp` |
| The per-op default | Everything else; the loosest node of the graph wins, and the same value is used for `atol` and `rtol` | `TestTolerances.hpp` through `src/harness/tolerance/ToleranceResolver.hpp` |

TOML overrides describe how far one engine may drift. They are never applied
when golden data is checked against a reference executor, and neither is the
dynamic tolerance: only the engine's output is graded with it, and golden data
is checked against a reference executor with the per-op default.

## The convolution-forward dynamic tolerance

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

### The model

For an output stored as half or bfloat16, computed with float accumulation,
and graded against a float-accumulating reference:

```text
|impl_i - ref_i| <= 2 * eps * |ref_i|                      rtol
                  + (sqrt(K) * u32 + lambda * eps) * sigma  atol
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
`K` of 6 or less. In the calibration data it also failed correct output (one
half and one bfloat16 output on gfx942, 18 on gfx90a).

The function is `calculateConvFpropOutputTolerance<T>()` in
`projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp`.

### What the harness feeds it

`convFwdErrorScale()` applies the model only when the graph has exactly one
`ConvolutionFwd` node, with float compute type, and its `x` and `w` are graph
inputs: not virtual, and not written by another node. A `MUL` node with one
operand from the convolution and the other a graph input (in either operand
order) scales `sigma` by that input's largest magnitude, and the scales of
several multiplies multiply; a `MUL` of anything else (the result with itself,
two inputs) leaves the model. The other pointwise modes are not examined: each
is assumed to change an error by at most its slope, about 1, so it leaves
`sigma` unchanged. That was checked against the shipped bundles only, which use
`ADD`, `MUL` and `RELU_FWD`; a mode with another gain, such as `DIV` or
`RECIPROCAL`, would be graded as if it had none. Nodes that are neither the
convolution nor pointwise are ignored. The scale is one value for the whole
graph, not one per output, and the shipped graphs have at most one multiply.
A graph outside what is described here, an input whose statistics are not
available (a NaN or an infinity in golden data or in a fill recipe, an
unreadable baked value) and every float output use the per-op default.

A graph input's statistics come from, in this order:

- the value baked into the graph, for a scalar constant (the `F27const` cases
  of the Variant3 bundles bake their scale as 0.5): `v^2` and `|v|`;
- the fill recipe, for a tensor the harness fills: `(lo^2 + lo*hi + hi^2) / 3`
  for a uniform draw on `[lo, hi]`, `v^2` for a fixed value, so no tensor is
  read;
- a scan on the host, for an input loaded from golden blobs, which is
  already there.

`sigma` is the typical output only for inputs with a zero mean. Then partial
sums cancel and the output grows as `sqrt(K)`. For a nonzero mean (a constant
fill, or uniform on `[0, a]`) the output is larger than `sigma` (the root mean
square is 3, 25 and 190 times `sigma` for uniform on `[0, 1]` at `K` = 16,
1,152 and 65,536), so `atol` is a smaller fraction of it and the `rtol` term,
which scales with the output, carries the tolerance. The calibration covered
the fills of the shipped bundles, which are mostly zero-mean.

### Calibration

Data: every half and bfloat16 `ConvolutionFwd` and `ConvolutionFwdPointwise`
bundle case that MIOpen runs, with outputs up to 4 M elements. Each output was
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

The scaled convolution graphs (`ConvolutionFwdPointwise` Variant2 to Variant6)
have no engine run in this table: on gfx942 the MIOpen engine ranks no engine
for any of their half and bfloat16 cases, so the 0.5 scale of Variant3 has been
graded only in host unit tests with mocked engine output. The first engine and
architecture that executes them should add the headroom of those outputs here.

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

### When an engine needs more

The model describes float accumulation and, for half, io-precision
intermediates of the size MIOpen shows. An engine whose kernels are less
accurate fails here first. Find the solver and the element that failed
before you add a `[[tolerance_overrides]]` entry for those cases, and
name the ticket that tracks the accuracy in its comment, as for any override.
