# Does the catalog win survive genuine 60-CU execution?

The campaign headline (**+23.9% wall-clock** for the widened TN HHS catalog) was measured with
60-CU **selection** (`--sm_count_target 60`) but 96-CU **execution**, because the CU-masked
stream was recorded as hanging ~37% of runs — too often for a 5 000-run sweep. That left the
central premise of the whole campaign ("benchmarking this card at 60 CUs approximates navi32")
*assumed rather than tested*.

It is now tested. **The win holds.**

## Result

207 shapes (stratified subsample of the 998-shape eval set, all 109 strata represented),
3 arms x 2 reps, **genuine CU-masked execution** — `hipExtStreamCreateWithCUMask`, 30 of 48
WGPs = 60 of 96 CUs, verified live by throughput (3 237 vs 4 958 GFlop/s = 65.3% against an
ideal 62.5%).

| arm | geomean | wall-clock |
|---|---|---|
| `ship_aa` (A/A control) | 100.12% | **99.89%** |
| `wide` (298-solution catalog) | **125.22%** | **122.71%** |

**A/A floor 0.11 pt** — the tightest of any arm comparison in this campaign, so the +22.7% is
roughly 200x the noise.

The jackknife is *favourable*: dropping the largest time consumers **raises** the win
(122.7% -> 124.6% -> 127.6% -> 136.2% after dropping 5/10/25/50), so it is not carried by a
handful of big shapes. The quoted figure is the conservative end.

| by size | wall-clock | | by geometry | wall-clock |
|---|---|---|---|---|
| large (37) | 121.4% | | **gemv (11)** | **246.9%** |
| medium (67) | 124.8% | | skinny (63) | 129.2% |
| small (53) | 126.8% | | rect (67) | 123.5% |
| tiny (49) | 131.1% | | square (65) | 121.1% |

Same shape of result as the 96-CU sweep: gains concentrate on GEMV and small shapes, exactly
where a 471-row nearest-neighbour table serves a distant neighbour.

## Was 96-CU execution a fair proxy? Yes — with a caveat

Restricting both regimes to the **same 206 shapes**:

| regime | geomean | wall-clock | its own A/A |
|---|---|---|---|
| 96-CU execution (the shipped measurement) | 125.6% | 125.7% | 102.1% |
| **60-CU execution (this run)** | **125.7%** | **122.7%** | **99.9%** |
| difference | **+0.1 pt** | **-2.9 pt** | |

**Geomean is identical.** Wall-clock is 2.9 pt lower under real 60-CU execution — and note
which measurement is the shakier one: the 96-CU sweep's *own* A/A control sat at **102.1%**,
i.e. that run carried ~2 pt of arm-position drift, while this one carries 0.11 pt. Most of the
2.9 pt gap is plausibly that drift rather than a genuine regime effect.

Mechanically the residual makes sense too: **large shapes gain least (121.4%) and dominate
wall-clock**, so any regime change that slightly favours large shapes moves the time-weighted
metric while leaving the per-shape geomean untouched. That is precisely the pattern seen.

**Conclusion: the emulation shortcut was sound, and the shipped +23.9% is if anything
marginally optimistic.** Under genuine 60-CU execution the same comparison gives **+22.7%** on
a stratified subsample. The claim does not need retracting; it needs a ~1 pt haircut and this
citation.

## Correction to the runbook: the hang rate is 2%, not 37%

The runbook says the masked stream is "unusable for long sweeps" on the strength of **3
timeouts in 8 runs**. Over **1 242 masked runs** the real rate is **2.0% (25 timeouts)**, all
recovered by the harness's timeout + `pkill`.

An 8-run sample cannot distinguish 37% from 2%; the 95% CI on 3/8 runs down to about 8%. The
first masked run after an idle GPU *does* reliably hang, which is what an 8-run probe
oversamples.

**So execution fidelity is affordable after all** — a full 998-shape masked sweep costs roughly
3 hours, not the impossible budget assumed. Future campaigns on this card should use it rather
than fall back to selection fidelity.

## Reproduce

```bash
python3 bench_arms.py \
  --arms ship=$HOME/navi32/libs/navi32ship/library/gfx1100 \
         wide=$HOME/navi32/libs/wgm8/library/gfx1100 \
         ship_aa=$HOME/navi32/libs/navi32ship/library/gfx1100 \
  --shapes state/eval_shapes_masked.json --out results/P12_masked60.csv \
  --reps 2 --cus 60 --fixed-iters 20 --timeout 45      # masked is the DEFAULT
python3 analyze.py results/P12_masked60.csv ship
python3 compare_masked.py
```

**`HIPBLASLT_TENSILE_LIBPATH` must point at the arch subdirectory** (`.../library/gfx1100`),
not the arm root and not `.../library`. Pointing one level up produces `status=error` on every
single row at full speed — rows appear at the normal rate and every one is empty. Check
`status` counts before reading any number.
