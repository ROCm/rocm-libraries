# What the navi33 (gfx1102) question actually is — settled statically

The [README](README.md) and the campaign notes said the navi33 port was blocked on
*occupancy*, because navi33 has a **1024-VGPR/SIMD** register file against navi31/navi32's
**1536**, and that "the performance question needs real navi33 silicon".

That framing was too coarse. The occupancy exposure is computable from the built code
objects, and doing so narrows the open question considerably — **and contradicts the
implication that porting is what puts occupancy at risk.**

## Method

RDNA3 wave32 occupancy, from `rocisa/rocisa/include/hardware_caps.hpp`:

```
PhysicalMaxVgprCU = 2 (SIMDs) * vgprPerSimd * 32 (lanes)     # line ~655
MaxWavesPerSimd   = 16   for gfx11                            # line ~576
=> waves/SIMD = min(16, vgprPerSimd // align8(vgpr_count))
   vgprPerSimd = 1536 (gfx1100/1101/1151), 1024 (all other gfx11, incl. gfx1102)
```

`vgpr_count` is read per kernel from the `.co` (`llvm-readelf --notes`), joined to the
winning kernel of each measured shape.

> **Do not use `Tensile/OccupancyMeasure.py` for this.** It is explicitly
> gfx9/wave64-specific and `_arch_caps_for_kernel` returns `None` for gfx11. It will not
> error — it declines — and a caller that ignores the `None` gets a confident wrong number.

## Result — the exposure is real, and it is not caused by porting

Fraction of each catalog whose kernels drop waves/SIMD on navi33:

| catalog | kernels | median VGPR | >=252 VGPR | lose waves on navi33 |
|---|---|---|---|---|
| **navi32 SHIPPED (HHS)** | 60 | 256 | 37 | **100.0%** |
| navi31 port (HHS, this branch) | 238 | 256 | 189 | **98.3%** |
| navi32 shipped (BBS) | 55 | 256 | 33 | **100.0%** |
| navi31 port (BBS) | 246 | 256 | 197 | **98.4%** |

Weighted by measured kernel time across all four ported ProblemTypes: **100.0% of time**
is spent in solutions that lose occupancy on navi33. The dominant mode is
**256 VGPR: 6 -> 4 waves/SIMD (-33%)**; 96-VGPR kernels take the worst relative hit
(16 -> 10, -38%) because they sit exactly at the 16-wave cap on a 1536 file.

**The control is the point.** navi32's *own shipped* catalog is **100%** exposed — the
port is marginally *less* so (98.3%), and is the only one of the two containing any
kernels at <=64 VGPR (4 of them), which keep 16 waves on both parts. So the 33% cut is a
property of these TN GEMM tile shapes on RDNA3, **not something the port introduces**.
Whatever navi33 ships today already pays it.

## What this changes

- **navi32 (gfx1101) is unaffected** — identical 1536-VGPR file, so occupancy is
  bit-identical to navi31. This is a retroactive validation of the port that shipped on
  this branch: it costs *zero* occupancy, which is why the build gate found no
  `overflowedResources`.
- **The navi33 blocker is narrower than stated.** It is *not* "will it fit" (it does —
  `MaxVgpr` is 256 on all RDNA3), and *not* "will it lose occupancy against navi33's
  baseline" (both sides lose equally). The remaining question is:

  > **do scheduling parameters tuned at 6 waves/SIMD still hold at 4?**

  `PrefetchGlobalRead`, `PrefetchLocalRead` and depth choices are occupancy-sensitive,
  and every ported kernel would run at 2/3 the wave count it was tuned at.

- That question still needs navi33 silicon — but it is a *retune-the-schedule* question,
  not a *port-is-invalid* question, and it applies equally to navi33's existing catalog.

## Reproduce

```bash
python3 ~/navi32/navi33_occupancy.py     # joins .co VGPR counts to measured winners
```

The script asserts a non-empty join. The first version of it silently reported **0.0%
exposure** for every ProblemType because the kernel-name join matched nothing — bench
kernel names carry host-side tokens (`GSU1`, `WGM8`, `SU32`) absent from the compiled
symbol, and splitting on `_` shreds multi-part tokens like `MIWT1_3`. Matching is by
token-set subset after rejoining numeric fragments (157/157 unique). **A zero-match join
reads exactly like a clean negative result** — hence the assert.
