# What the navi33 (gfx1102) question actually is — settled statically

> **This document was published on 2026-08-24 with a wrong headline number (100% of kernel
> time exposed) and corrected the same day to 10.1%.** The error and its cause are recorded
> at the bottom, because the mistake is more instructive than the result.

The campaign notes said a navi33 port was blocked on *occupancy*, since navi33 has a
**1024-VGPR/SIMD** register file against navi31/navi32's **1536**, and that settling it
"needs real navi33 silicon".

The exposure half is computable from the built code objects. Doing so shows the concern is
**much smaller than assumed**: only **10.1% of measured kernel time** runs in solutions that
lose occupancy on navi33.

## Why: LDS binds, not VGPRs

RDNA3 wave32 occupancy is a **minimum over several limits**:

```
waves/SIMD from VGPR = min(16, vgprPerSimd // align8(vgpr_count))
WGs/CU    from LDS   = DeviceLDS(65536) // roundup256(group_segment_fixed_size)
effective WGs/CU     = max(1, min(vgpr_limit, lds_limit))
```
(`rocisa/rocisa/include/hardware_caps.hpp`: `PhysicalMaxVgprCU = 2*vgprPerSimd*32`,
`MaxWavesPerSimd = 16` for gfx11, `DeviceLDS = 65536`.)

For these TN GEMM kernels **LDS is what binds, overwhelmingly**:

| limiter on navi31/navi32 | kernels |
|---|---|
| **LDS** | **869** (88%) |
| VGPR | 56 |
| tie | 62 |

A kernel using ~65 KB of LDS fits **one workgroup per CU no matter how many VGPRs it uses** —
so shrinking the register file by a third changes nothing for it. Only the low-LDS kernels
are VGPR-limited, and only those lose waves.

Worked example of each case:

| | LDS | thr | VGPR | navi32 | navi33 | |
|---|---|---|---|---|---|---|
| high-LDS | 65 024 | 256 | 256 | 1 wg / 8 waves | 1 wg / 8 waves | **same** |
| low-LDS | 3 776 | 128 | 256 | 3 wg / 12 waves | 2 wg / 8 waves | **loses 33%** |

## Result

Fraction of measured kernel time in solutions that lose waves/SIMD on navi33:

| ProblemType | kernels | LDS-bound | % TIME loses | % SHAPES loses |
|---|---|---|---|---|
| HHS | 238 | 212 | **10.7%** | 6.3% |
| BBS | 246 | 219 | **7.1%** | 4.0% |
| AuxH | 250 | 219 | **11.0%** | 7.0% |
| AuxB | 253 | 219 | **11.2%** | 7.0% |
| **all** | | | **10.1%** | |

Where it does bite the drop is real — `v=167, lds=13824: 16 -> 12 waves/CU` and
`v~255, lds~15616: 12 -> 8 waves/CU` — but it reaches only about a tenth of the work.

## What this changes

- **navi32 (gfx1101) is unaffected either way** — identical 1536-VGPR file, so occupancy is
  bit-identical to navi31. The port that shipped on this branch costs *zero* occupancy, which
  is why the build gate found no `overflowedResources`.
- **The navi33 occupancy objection largely dissolves.** It is not "will it fit" (it does —
  `MaxVgpr` is 256 on all RDNA3), and it is not a broad occupancy cliff (88% of kernels are
  LDS-bound and indifferent to the register file). The residual risk is confined to the ~10%
  of time in VGPR-limited kernels.
- Still unmeasured, and still needing silicon: navi33's **CU count and memory system** differ
  too, and whether schedule params hold for that ~10%. But the register-file argument — the
  one reason previously given for treating navi33 as out of reach — is now quantified, and small.

## Reproduce

```bash
python3 ~/navi32/navi33_occupancy.py     # occ_lib.py holds the occupancy model
```

## Two traps, and the one that actually got me

**`Tensile/OccupancyMeasure.py` must not be used here.** It is explicitly gfx9/wave64-specific
and `_arch_caps_for_kernel` returns `None` for gfx11. It *declines* rather than errors, so a
caller that ignores the `None` gets a confident wrong number.

**A zero-match join reads exactly like a clean negative result.** The first version of the
script matched no kernels at all and printed `0.0% exposure` for every ProblemType; only a
downstream ZeroDivisionError exposed it. Bench kernel names carry host-side tokens (`GSU1`,
`WGM8`, `SU32`) absent from the compiled symbol, and splitting on `_` shreds multi-part tokens
like `MIWT1_3`. Matching is now by token-set subset after rejoining numeric fragments
(157/157 unique), and the script **asserts a non-empty join**.

**And the one that reached a published number: I computed one term of a `min()` and reported
it as the answer.** Occupancy is `min(vgpr_limit, lds_limit, 16)`. The wiki flags navi33's
VGPR file, so the VGPR term is the one I computed — and by that term alone 98–100% of kernels
"lose occupancy". But it is not the binding term for 88% of these kernels, so the number
described nothing real. The correct figure is 10.1%.

> **When a quantity is a minimum over several constraints, computing one term tells you
> nothing until you have shown it is the term that binds.** The failure was not arithmetic —
> every number in the first version was correct — it was answering a different question than
> the one asked, in units that looked identical to the right answer.
