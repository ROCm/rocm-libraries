# Scheduled FMHA accuracy checks

The forward tests compare decoded FP16/BF16 inputs with an independent FP32
QK, online softmax and PV implementation. The reference does not narrow P to
the device operand type. O and LSE are never replaced with device results.

## Strict diagnostics and dtype contract

`Compare` without an `AccuracyContract` keeps the original strict behavior:
max absolute O error <= 0.015625, minimum row cosine >= 0.99998, mean row cosine
>= 0.99999, and the existing finiteness/LSE checks. With an explicit contract,
`strict_passed` and all cosine metrics are still calculated and reported.
BF16 forward tests gate on the component contract; FP16 forward tests require
both the component contract and the strict check. Historical strict failures
are preserved in the external PR validation evidence.
Both 16-bit contracts retain the original absolute O error cap of 0.015625. A large
contribution budget cannot override that cap. The FP8 contract ("FP8 P" below) caps each
component by its own limit instead.

The cosine constants originated in the local FMHA tests before this test
migration. They are not an upstream universal BF16 attention accuracy guarantee.
For comparison, the stock FMHA example uses dtype-specific elementwise
limits and a narrowed P/O reference. Passing that check is not the same claim
as passing this independent FP32 comparison.

## Component budget

For each selected output component, the reference also computes
`A[d] = sum_k p[k] * abs(V[k,d])`, with unquantized probabilities and the same
normalization, masks, descales and optional V=0 virtual sink as the O reference.
The additional accumulator uses double precision; it does not change the FP32
O/LSE calculation. Let `r` be FP32 reference O, `K` the number of valid keys,
`Td = ceil(key_length / key_tile_size)`,
`Tr = ceil(key_length / reference.key_chunk_size)`, and `u32 = 2^-24`.
The contract accepts N64/N128. `Compute` records its actual key chunk size so
the budget counts both device and reference online reductions, including host
tests that use smaller chunks. The double A accumulator
uses the FP32 oracle probabilities; it is not an independent FP64 softmax.

The test uses FP16 unit roundoff `u = 2^-11` or BF16 `u = 2^-8`, and
`gamma(n) = n*u32 / (1 - n*u32)`. BF16 `u = 2^-8` assumes round-to-nearest
narrowing of device P and O. The tested gfx1250 path uses ties-to-even
conversion, including the LLVM builtin BF16 cast when enabled. That builtin
path does not consult `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT`; the effective
converter must be checked in the final build. A path that actually truncates
P or O needs a `u = 2^-7` rounding bound and separate contract validation; it
is outside this tested budget. For the tested FP32 score path:

```text
eta  = max_k (abs(scale) * gamma(Dq) * sum_d abs(Q[d]*K[k,d])
              + u32 * abs(score[k]))
beta = 2*gamma(K) + gamma(3*Td + 8) + gamma(3*Tr + 8)
       + expm1(4*eta) + 4*u32
E    = beta*A + gamma(2)*abs(r) + u*A + floor*K*max_k(abs(V[k,d]))
limit = (1 + u)*E + u*abs(r) + floor
```

The factors of two account for comparing the device accumulation with a
separate FP32 reference, rather than treating either as exact. Score differences
between the two paths can reach `2*eta`; normalization adds another factor of
two in the probability ratio bound, giving `expm1(4*eta)`. The two chunk terms count their separate
online rescale/reduction operations. The final
`(1+u)` retains the cross term from rounding O after P and FP32 error.
`floor` is 2^-25 for FP16 round-to-nearest subnormal spacing, and 2^-126
for the BF16 path that permits flushing subnormals to zero. FP16 subnormal
behavior must be verified for the selected runtime; the floor does not assert
that every hardware/compiler path preserves FP16 subnormals.

A component passes when `abs(actual-r) <= limit` and the original absolute
error cap is satisfied. Nonfinite O always fails.
LSE keeps `atol=0.02`, `rtol=0.002`; -infinity is allowed only on empty rows
without a finite virtual sink. The comparison reports the maximum component
ratio, failing element/row counts, global relative L2, worst row relative L2,
row norm drift, per-head projected gain and the original cosine results.

This is a **tested numerical budget**, not an operation-by-operation proof of
all hardware implementations. In particular, `4*u32` for exp accuracy, folded
log2 scale/FMA differences, and the online reduction allowance are explicit
policy assumptions. The supported validation domain consists of the finite,
bounded decoded inputs used by these tests. Overflow, nonfinite budgets and
invalid metadata fail rather than creating an infinite tolerance. Wide score
ranges and underflow require additional evidence; do not infer their behavior
from the bounded-input tests.
The fixed absolute cap is also specific to this bounded test domain: for
example, BF16 output rounding alone can reach 0.015625 at `abs(O)=4`.
It is not a universal guarantee for arbitrary magnitudes of V or descales.
The formal properties use `abs(V)<=1.75`; the full performance matrix input
range is checked separately before applying this contract.

For FP16, the per-key floor is deliberately conservative even when a
probability is normal. At K=32768 it contributes at most
`32768*2^-25*max(abs(V))` before the final O rounding; this can dominate
the dtype rounding term. FP16 still requires the original strict check.
Conversion/WMMA subnormal handling requires native controls and code-object
evidence; a compiler flush flag alone is insufficient evidence.

## FP8 P (fp8bf16)

The FP8 page tile (`fp8bf16`: FP8 Q/K/V/P, BF16 O) converts `kPScale * P` to OCP e4m3, with P
taken against its running row maximum, and divides O by the mass of the converted P (the
quantized row sum; no LSE, no sink). Its contract is `AccuracyContract{7, key_tile,
QuantizedP{...}}`; the reference must be computed with the same `QuantizedP`
(`Compute(sequences, chunk, quantized_p)`), otherwise `Compare` refuses it. The model has no sink
term and no LSE: `Compare` also refuses an FP8 contract on a sequence with a finite sink logit or
on an output with LSE.

Per key the converted P' is either `P (1 + d)` with `abs(d) <= u_P = 2^-4` (round to nearest even
in the normal range) or `P + e` with `abs(e) <= min(P, loss)` once `kPScale * P` is below e4m3's
smallest normal `2^-6`, that is `P < t0 = 2^-6 / kPScale` (`2^-14` for `kPScale = 256`). `loss` is
half a subnormal step (`2^-10 / kPScale = 2^-18`) when the convert keeps e4m3 subnormals and `t0`
when it flushes them. Round to nearest even rounds some subnormal keys up (`e > 0`); the
numerator below needs only `abs(e)`, and the denominator's lower bound only the downward errors.
With `L` the exact normalizer:

```text
O - r = sum_k (P'_k - P_k) (V_k - r) / sum_k P'_k
S[d]  = sum_{k: P_k < t} min(P_k, loss) * abs(V[k,d]) / L
S0    = sum_{k: P_k < t} min(P_k, loss) / L,          t = t0 * exp(4*eta)
P_fp8 = (u_P * (A + abs(r)) + S[d] + S0 * abs(r)) / (1 - u_P - S0)
E     = beta*A + gamma(2)*abs(r) + P_fp8
limit = (1 + u_O)*E + u_O*abs(r) + 2^-126,            u_O = 2^-8 (BF16 O)
```

- The normalization term (`A + abs(r)` rather than `A`) is there because the kernel divides by the
  converted mass: on one key of mass `m < 1/2` and value 1 among zeros, `sum_k P_k abs(V_k - r)`
  is `2m(1 - m)`, more than `A = m`.
- The floor counts only the keys that can be subnormal on the device. The device takes P against a
  running maximum that never exceeds the final one, so its P is at least the reference's; the
  score difference between the two paths widens the threshold by `exp(4*eta)`.
- The floor is mass-weighted. A key loses at most its own mass, so the floor does not grow with the
  number of small keys. An unweighted `floor * K * max(abs(V))` with the flushed floor `2^-14`
  reaches `2 * max(abs(V))` at K = 32768, and with a score standard deviation near 2 an estimated
  (Gaussian scores) few hundred keys per row at K = 3000, and thousands at K = 32768, are below
  the threshold.
- `keeps_subnormals` is set from native controls of the P convert
  (`__builtin_amdgcn_cvt_scalef32_pk8_fp8_f32`), of the row-sum dequant and of the FP8 WMMA's
  subnormal inputs; a compiler flush flag alone is not evidence. On gfx1250, with the page-tile
  gtests' `-fgpu-flush-denormals-to-zero`, all three keep e4m3 subnormals and the convert rounds
  to nearest even (`test_batch_prefill_fp8_native_controls`), so the FP8 tests use
  `keeps_subnormals = true` (`loss = 2^-18`).

The fixed absolute O cap of 0.015625 does not apply to FP8 contracts: correct FP8 output on rows
with a few keys exceeds it. In `QuantizedPContract`, a row of two keys with V = -1 and +1, whose
second `256 * P` (136.5) converts to 144, misses the FP32 reference by 0.025 within its limit of
0.088. Each component is capped by its own limit instead, `Metrics` reports the largest limit
(`max_component_limit`), and the final gate is the component contract (`RequiresStrict()` is
false).

The host tests model the device arithmetic on the CPU (per 128-key tile: FP32 scores with the
descales folded in, P against the running maximum, `kPScale * P` rounded onto the e4m3 grid with
subnormals kept or flushed, FP32 accumulation of O and of the converted mass with the online
rescale, `O / mass * v_descale` rounded to BF16). `Fp8ModelPassesContract` runs it from 2 to 73728
keys. `Fp8ControlsFireAtLongK` requires the negative controls to fail through the component ratio,
with subnormals kept and flushed, while the unperturbed model passes on the same data:

- v descale x1.25 and one wrong page-table entry on SparseMarker V (V = 1 on five spread 64-key
  pages, 0 elsewhere), at K = 3000 (the gtests' Window long rows), 8192, 32768 (Deterministic) and
  73728 (the longest chunk8k row: a 64K prefix and an 8K chunk);
- q and k descale x1.5 on scores with a standard deviation near 2 and random V, at the same K;
- q and k descale x1.25 on that data up to K = 8192. From K = 32768 on, random V averages a 25 %
  score-scale error into the budget (measured ratios 0.85-1.03 at K = 32768 and 73728): this is
  the contract's detection limit for that error on that data, not a pass criterion.

Random V does not serve the v and page controls: their effects shrink with K (one page carries
about 64/K of the mass; v x1.25 moves O by 0.25 * abs(r), and abs(r) falls like
`sigma_V / sqrt(K_eff)`) while the P term does not.

## Gain and cancellation controls

A large `A/abs(r)` represents cancellation. Consequently neither a global
relative L2 bound of `uP+uO` nor an averaging-based gain threshold is a universal
property of signed attention inputs. Deterministic rounding may be correlated
across identical rows. Those metrics remain diagnostics and are compared with
same-input stock `qr_tdm` controls in the PR evidence.

The formal generated-dispatch tests run each of their 40 shapes with signed,
constant and nonnegative V, with LSE off/on, for FP16 and BF16: 240 tests per
dtype, 480 total. This covers the existing batch/group, layouts, masks, tails,
virtual sinks and selector boundaries without duplicating kernel geometry.
Constant channels use several exactly representable signed mantissas based on
`{0.75, -1.25, 1.5, -1.75}`, with head variation. Without a virtual sink, exact
arithmetic returns the constant on each nonempty attention row, independent
of QK. The device result is checked against the component budget; in particular,
FP16 subnormal P rounding at extreme score differences can cause a deviation
from the exact constant. Empty rows return zero; a finite virtual sink with V=0
reduces the constant by the probability mass assigned to real keys, which the
independent oracle computes.
For this no-cancellation property, the component budget also detects gain
errors. Several mantissas are needed: BF16(1.01) equals 1.0078125 and an
all-ones V alone would not reject that post-rounded gain.

Gain detection also depends on K and the score budget `eta`. The host
`LargestKeyGainSensitivity` control covers K=32768, Dq=192, N64/N128, with
every Q/K component equal to 0.125 and the signed constant channels above.
It rejects both post-BF16-rounded +/-1% gains. This is a specific property
control, not a guarantee for larger score roundoff or arbitrary QK inputs.
CPU geometry controls with concentrated tail/boundary weights include
inputs where a 1% gain remains inside the component budget. Small missing
tile contributions and mild rescale errors can also remain inside it.
Those measured sensitivity limits are preserved in the external evidence.

Host unit tests verify exact outputs, cancellation with repeated rows,
post-rounded gain, a single over-budget component, the absolute error cap,
zero outputs, NaN, LSE mismatches and invalid contribution metadata.
External CPU fault models additionally check skipped PV contributions, stale
rescale and causal boundary faults. Same-input native captures measure the
actual kernel outputs separately. A CPU fault model pass is not a native
regression test pass.

The contract does not establish end-to-end model quality. Full PR validation,
FP16/BF16 coverage, same-input stock controls, the largest-K fault checks and
reviewer acceptance remain separate gates.
