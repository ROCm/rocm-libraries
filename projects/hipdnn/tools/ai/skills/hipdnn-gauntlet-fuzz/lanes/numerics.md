# Lane: numerics and conditioning (`NUM`)

**Trigger:** always. **Agent:** `scout`. **Output:** `R/lanes/NUM.md`: the bound, then ideas.

Two jobs: find the precision regimes that stress the kernels, and state the bound every graded space must keep so the oracle can
judge it.

## Method

1. Start from brief section 5. For each graded space, compute the largest value of the quantity that breaks judgeability (SDPA
   backward: `|scale| x dqk x mag^2 (+ |bias|)`, `mag` = the largest |Q|, |K| of the space's input domains) and compare with
   `atol/10 x 2^24` per dtype (168 fp32, 8388 fp16, 16777 bf16 at the gauntlet's flat tolerances). Name every axis value that breaks it.
2. Amplifiers: `1/(1-p)` near p = 1, large scales, large bias, long reductions, saved statistics fed back.
3. Hardware number formats per arch: subnormal flush, denormal modes, fp8 ranges.
4. Exact-zero outputs (a gradient that is exactly zero, a single-key row) where a tiny engine error fails a relative test.

## Output

First a table: space, axis values that break the bound, the bound, the proposed constraint (as a `derive`/`constraints` expression
the generator can encode). Then ideas for stressing precision with judgeable inputs: for an extreme parameter, the inputs that keep it
judgeable (scale 2^20 with Q and K magnitudes in [5e-4, 1e-3]).
