# The tier-0 score

The cycles per main-loop trip, per wave, that tier 0 ranks candidates by. Code:
`search._tier0`, `perf.orderCore`, `perf.sideCost`.

```text
score = max(C_wmma, C_lds) + exposed_A + exposed_B + tdm + tdmExposed + tripOverhead + icache
```

| Term | How it is computed |
|---|---|
| `C_wmma` | WMMAs per trip × `wmmaCycles` (8 for MXF8 16×16×128) |
| `C_lds` | ds_loads per trip × `dsCyclesPerLoad` (6) |
| `exposed_A`, `exposed_B` | the ds_load latency each side leaves exposed: [exposed-ds-latency.md](exposed-ds-latency.md) |
| `tdm` | tensor loads per trip (`perf.tensorLoads`) × `tdmIssueCycles` (40) |
| `tdmExposed` | `perf.tdmExposed`: `tdmLatency` (800) for a one-copy body, else `max(0, tdmLatency − max(C_wmma, C_lds))` |
| `tripOverhead` | 17: loop branch, barrier signal/wait, waitcnts |
| `icache` | `max(0, body − icacheBodyInstructions) / copies × 0.25`, with `body = (WMMAs + ds_loads) × copies` |

`C_wmma` and `C_lds` depend on the loop order only; the register choice changes the exposure,
the steady-body copies and the VGPRs.

- **ds_loads per trip** (`orderCore`): per side, resident tiles × ds_loads per tile, plus
  resident tiles / 2 for the MX scales.
- **copies** is the LCM of the two sides' copies; a side's copies are `VA / gcd(VA, positions)`,
  or 1 under CLR.

Ties go to fewer units loaded per trip (each pays the latency), then fewer copies, then fewer
VGPRs, then the canonical order.

## Worked example: the tuned MXF8 kernel

`LoopOrder` `KKNMNM`, one trip `[Ki 2, No 2, Mo 2, Ni 5, Mi 4]`:

- `C_wmma` = 160 WMMAs × 8 = 1280.
- `C_lds` = 162 loads × 6 = 972: A 16 tiles × 4 plus B 20 tiles × 4 is 144 loads, plus 8 + 10 = 18
  scale loads.
- `exposed_A` = `exposed_B` = 0 (A PLR 2 / VA 3, B PLR 1 / VA 2; see the exposure note).
- `tdm` = 2 tensor loads (A with MXSA, B with MXSB) × 40 = 80.
- `tdmExposed` = 0: the body has 3 copies, and one 1280-cycle trip covers the 800.
- `icache` = 0.

```text
score = max(1280, 972) + 0 + 0 + 80 + 0 + 17 + 0 = 1377
```

The trip is WMMA-bound, and the only register choices that change the score are the ones that
expose latency; A with PLR 1 / VA 2 exposes 560 and scores 1937.  A with CLR 1 / VA 4 exposes
nothing but makes a one-copy body, so it scores 1377 + 800 = 2177.

## Why a one-copy body pays the whole tensor-load latency

`StinkyWaitCntInsertionPass` sets the `s_wait_tensorcnt` depth on the scheduled loop.  A one-copy
steady body waits for `tensorcnt 0` every trip, 0.04 to 1.0 trips (median about 0.3) after the
load it retires issued; a multi-copy body keeps each load in flight 1.0 to 1.25 trips.  On 6
variants measured twice on one gfx1250 card (2026-11-19, 4096×5120×65536, 14-24 kernels each,
the faster of the two runs kept), one-copy kernels run at a median 70-79% of the variant's best
and multi-copy ones at 87-96%, on every 8×10 / 8×8 wave tile:

| Variant (wave tile, DepthU) | Tier 0 before | Tier 0 now | Tier 2 | Tier 3 |
|---|---|---|---|---|
| MXF8 16×16×128 8×10, 256 | +0.46 | +0.52 | +0.69 | +0.94 |
| F8 16×16×128 8×10, 256 | +0.82 | +0.81 | +0.78 | +0.98 |
| MXF8 16×16×128 4×4, 256 | +0.78 | +0.78 | +0.69 | +0.37 |
| F8 16×16×64 8×10, 128 | −0.04 | +0.34 | +0.13 | +0.68 |
| BF16 16×16×32 8×8, 64 | −0.31 | +0.62 | +0.62 | +0.82 |
| BF16 16×16×32 4×4, 64 | −0.56 | −0.63 | −0.66 | −0.18 |

(Spearman against GFLOPS within each variant.)  The 4×4 tiles' trips (256 cycles) are far
shorter than the load latency, so every body drains whatever its copies.  The term still
charges a one-copy body `tdmLatency` and a multi-copy one `tdmLatency − 256`, 800 against 544,
and on BF16 4×4 that is backwards: the one-copy kernels with both operands resident (CLR 1)
measured 0.97 PF on 2026-11-20, against 0.79 for every multi-copy kernel tied at the best
score.  They score 1201 against 997, 20% above the best, so a tuning set drawn near the best
score never holds them; tier 2 ranks them first (408–422 cycles against 832).  See
[selection.md](selection.md).  The term should not charge a short trip's one-copy body more
than its multi-copy body; that change is not made yet.  BF16 4×4's slowest kernels also hold all of A
and B's next-trip ds_loads in registers past v255 and switch `s_set_vgpr_msb` inside the WMMA
chain, which no tier models.  The 40-kernel MXF8 calibration set (all but one multi-copy)
stays at +0.51.
