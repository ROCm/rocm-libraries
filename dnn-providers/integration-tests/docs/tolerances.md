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
when golden data is checked against a reference executor.

## The convolution-forward dynamic tolerance

### Why a fixed number does not work

The per-op default for half and bfloat16 convolution forward is 0.1 for both
`atol` and `rtol`: about 10 % of a typical output at every size. That is
10 to 25 bfloat16 rounding steps, and over 100 half steps.

The older bound in the test SDK, `calculateConvFpropTolerance()`, scales with
the problem, but assumes an output can be as large as
`K * max|x| * max|w|`, where `K` is the number of products summed per output.
With inputs of both signs a real output grows only as `sqrt(K)`. Relative to
the output the bound therefore grows as `sqrt(K)`: for bfloat16 it is wider
than a typical output from `K` of about 1,300 on, and passes almost anything.

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
| `sigma` | The typical output magnitude, `sqrt(K * mean(x^2) * mean(w^2))`, times the largest magnitude of any tensor a fused multiply applies. The mean squares come from how the inputs were filled (see below). |
| `eps` | Machine epsilon of the output type: 2^-7 for bfloat16, 2^-10 for half. |
| `u32` | Unit roundoff of float, 2^-24. |
| `lambda` | 0 for bfloat16, 12 for half (calibrated, see below). |
| `minNormal` | The smallest normal number of the output type. |

Each term is one way two correct implementations differ:

- **Output rounding** (`2 * eps * |ref_i|`). Both sides round their float
  result once to the output type. Two correct roundings of nearly equal values
  differ by at most one unit in the last place, and `eps * |ref_i|` is one to
  two of those units.
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
fraction of the output at every reduction length. For inputs uniform in
[-1, 1], the band at a typical output (`|ref| = sigma`), divided by `sigma`:

| Output | K | fixed 0.1 | older bound | dynamic |
|---|---|---|---|---|
| bfloat16 | 16 | 0.175 | 0.094 | 0.016 |
| bfloat16 | 1,152 | 0.109 | 0.82 | 0.016 |
| bfloat16 | 65,536 | 0.101 | 6.2 | 0.016 |
| half | 16 | 0.175 | 0.012 | 0.014 |
| half | 1,152 | 0.109 | 0.13 | 0.014 |
| half | 65,536 | 0.101 | 0.95 | 0.014 |

The older bound is tighter only for half at very short reductions, where it
also fails correct output (one half and one bfloat16 output on gfx942, 18 on
gfx90a).

The function is `calculateConvFpropOutputTolerance<T>()` in
`projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp`.

### What the harness feeds it

`convFwdErrorScale()` applies the model only when the graph has exactly one
`ConvolutionFwd` node, with float compute type, and its `x` and `w` are graph
inputs. A `MUL` node whose second operand is a graph input scales `sigma` by
that operand's largest magnitude. Other pointwise nodes (bias add,
activations) change an error by at most their slope, which is about 1 for
every supported activation, so they leave `sigma` unchanged. Any other graph,
and every float output, uses the per-op default.

The mean squares cost nothing for inputs the harness fills: they come from the
fill recipe (`(lo^2 + lo*hi + hi^2) / 3` for a uniform draw on `[lo, hi]`, `v^2`
for a fixed value), so no tensor is read. Inputs loaded from golden blobs are
already on the host and are scanned once.

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
bfloat16 conv-fwd case MIOpen runs, at all four levels:

| Hardware | Input seed sets | Failures the dynamic tolerance adds |
|---|---|---|
| MI325X (gfx942) | 4 | 0 |
| MI250 (gfx90a) | 3 | 0 |

Every failure in those runs is an output MIOpen computes wrongly (values with
the wrong sign), which also fails the fixed 0.1. One correct gfx90a half output
that the fixed 0.1 failed (one element 0.12 off at `|ref|` of about 0.1, with
`K` = 18,432) passes.

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
apart from the kernels the `lambda` term allows for. For half at `K` above
16,000 the absolute term (`12 * eps * sigma`, about 0.5) is larger than the
fixed 0.1, so a missing filter tap there is caught in 29 % of outputs against
62 % for the fixed 0.1 (21 outputs); a missing input channel is still caught
in all of them.

### When an engine needs more

The model describes float accumulation and, for half, io-precision
intermediates of the size MIOpen shows. An engine whose kernels are less
accurate fails here first. Find the solver and the element that failed
before you add a `[[tolerance_overrides]]` entry for those cases, and
name the ticket that tracks the accuracy in its comment, as for any override.
