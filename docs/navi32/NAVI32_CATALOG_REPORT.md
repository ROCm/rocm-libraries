# navi32 (gfx1101) Origami + catalog — TN HHS

Developed on a Radeon RX 7900 XTX (gfx1100) configured to approximate navi32.
Branch `vmijovic/navi32`, worktree `~/navi32/src` off `origin/develop` @ `a9b7332a925`.

Written incrementally, one section per phase, so a crash preserves findings.

## Summary

**Shipped: navi32's TN HHS catalog widened from 73 to 298 solutions — +40% geomean,
+24% wall-clock**, on branch `vmijovic/navi32` (2 commits, pushed).

| what was tried | result |
|---|---|
| **Port navi31's catalog to navi32** | **+40% geomean / +24% wall-clock — SHIPPED** |
| Add gfx1101 to Origami (9 sites) | done; navi32 was previously unrecognised entirely |
| Re-fork `WorkGroupMapping` for 60 CUs | **null** — 6/8/10 within 0.33 pt of each other |
| Switch to an Origami Prediction library | **rejected** — 13 pt worse than GridBased |
| `ROC_GLOBAL_CU_MASK` to emulate 60 CUs | **cosmetic** — replaced with a per-stream mask |

The gain is concentrated where a sparse lookup table hurts most: **2.3–2.7x on tiny and GEMV
shapes**, ~1.2x on large square ones. All 298 solutions were gated through a real gfx1101
build (0 assembler errors, 0 VGPR overflows, ELF `Flags: 0x46, gfx1101`).

**Two well-motivated hypotheses were rejected by measurement**, and both are recorded so they
are not retried: WGM re-forking (mechanically real — 8 is ragged on 30 WGPs — but
unmeasurable), and Origami-Prediction selection (level with GridBased on large shapes,
collapsing on small ones). The `pred73` control isolates why: Origami over navi32's *own* 73
solutions gains nothing, so **the win is the catalog, not the selector**.

**Fidelity caveat that applies to every number here.** Selection is navi32-correct
(`--sm_count_target 60`), but execution is on all 96 CUs because the real CU mask hangs ~37%
of runs. Arm *ratios* are sound — the execution error is common-mode — but absolute
throughput is optimistic for navi32, and the memory system (960 GB/s / 96 MB vs navi32's
624 / 64) is not emulated at all.

---

---

## P0 — Infrastructure and the emulation gate

### The target

navi32 = RX 7800 XT = **60 CUs / 30 WGPs**, ISA `(11,0,1)`. This card is 96 CUs / 48 WGPs.

### What ships today (from `origin/develop`, ProblemType `Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV`)

| | navi31 | **navi32** |
|---|---|---|
| solutions | 298 | **73** |
| shape-table rows | 9 680 | **471** |
| `WorkGroupMapping` | all 8 | **all 8** |
| LibraryType | GridBased | GridBased |

navi32 has **4x fewer kernels and 20x sparser shape coverage** than navi31 on the same
ProblemType, and no Origami/Prediction path at all. That is the opportunity.

### RESULT: navi32 emulation works, and is accurate to 0.1 pt

`HIPBLASLT_BENCH_CU_MASK=60` gives **real 60-CU execution** on this 96-CU card:

| mask | reported | 4096³ TN HHS | ratio |
|---|---|---|---|
| `=96` | 48 of 48 WGPs = 96 CUs | 89 080 GF/s | 100% |
| **`=60`** | **30 of 48 WGPs = 60 CUs** | **55 556 GF/s** | **62.4%** |

Ideal is 60/96 = **62.5%**. The emulation reproduces it to within 0.1 pt.

Combined with `--sm_count_target 60` for the selector, both halves of the emulation are now
covered: correct kernel *choice* for a 60-CU part, and correct 60-CU *timing*.

### How that was nearly missed — two wrong turns worth recording

**Wrong turn 1: `ROC_GLOBAL_CU_MASK` looked like it worked, and does nothing.**

The plan's headline emulation mechanism **does not work**, and it fails in the most
dangerous way: it changes the number a naive query returns, so it looks like it works.

| probe | unmasked | masked (60 bits) | verdict |
|---|---|---|---|
| `hipGetDeviceProperties().multiProcessorCount` | 48 | **30** | changes — *this is the trap* |
| `TENSILE_STREAMK_DYNAMIC_GRID=0` grid (= `computeUnitCount` literally) | 48 | **48** | **unchanged** |
| launch grid, `DYNAMIC_GRID=6` (shipped default) | 183 | **183** | **unchanged** |
| launch grid, `DYNAMIC_GRID=7` (= N_CU) | 96 | **96** | **unchanged** |
| **throughput, 4096³ compute-bound** | 78 005 / 78 105 GF/s | **78 307 / 77 671** | **100.4% / 99.4% — unchanged** |

So the mask fools `hipGetDeviceProperties` but **neither restricts execution nor reaches
hipBLASLt's selector**, which reads its CU count from somewhere the mask does not touch.

Had the gate stopped at "multiProcessorCount says 30", the entire campaign would have been
built on a 96-CU card while believing it was measuring 60. The compute-bound throughput test
is what settles it: a real 60/96 restriction must show ~62.5%, and it shows 100%.

**Wrong turn 2 — and this one produced a wrong conclusion I published before catching it.**
On the strength of the table above I recorded "the mask is cosmetic". That was wrong, and the
reason is the same one that makes `ROC_GLOBAL_CU_MASK` look inert: **each bit of a HIP CU mask
selects a WGP, not a CU.** A 60-bit mask asks for 60 WGPs = 120 CUs, which on a 48-WGP part is
no restriction whatsoever — so it correctly measured 100%.

What exposed it was sweeping the mask size instead of testing one value:

| mask bits | GF/s | ÷ unmasked | consistent with |
|---|---|---|---|
| 16 | 30 766 | 34.1% | 32 CUs |
| 32 | 58 011 | 64.4% | 64 CUs |
| 60 | 83 687 | 92.9% | ~96 CUs (clamped) |

Every row is 2x the CU count the bit count implies. A single measurement at 60 bits is
indistinguishable from "inert"; the *slope* is what identifies the units. The lesson is
general: **when a knob appears to do nothing, sweep it before concluding it is inert** — an
inert knob and a mis-scaled knob look identical at one point.

`ROC_GLOBAL_CU_MASK` is still not usable here, but for the narrower reason that it does not
reach hipBLASLt's selector (`DYNAMIC_GRID=0`, whose grid *is* `computeUnitCount`, still reads
48 under it). The per-stream mask is what works.

**Operational hazard:** a mask requesting more WGPs than the device has does not error — it
**hangs the launch**. Hit once at 96 bits on a 48-WGP part; the process wedged at 1% GPU and
had to be `kill -9`ed. The patch now clamps and warns.

### What does work

| mechanism | reaches selector | restricts execution |
|---|---|---|
| `ROC_GLOBAL_CU_MASK` | no | no |
| **`--sm_count_target 60`** | **yes** — grid moves 183 → 60 (and → 30 at 30) | no |
| `hipExtStreamCreateWithCUMask` (`hip_runtime_api.h:2999`) | n/a | **yes, but needs a client patch** |

`--sm_count_target` reaches Origami's `num_cus` and the StreamK grid, so **selection is
correct for 60 CUs**. It does not restrict hardware, so **execution is still on 96 CUs**.

The bench client uses the default stream (`client.cpp:79`, `arg.streams = 0`), so a genuine
CU restriction requires creating a masked stream. That patch is to the *measurement harness*,
not the library under test, which keeps it methodologically clean.

### Protocol consequence

Emulation is done in two independent halves, and every result must say which it used:

- **selection fidelity** — `--sm_count_target 60`, correct kernel choice for a 60-CU part;
- **execution fidelity** — CU-masked stream, correct 60-CU timing.

Using only the first gives correct *choices* timed on the wrong machine.

---

## P1 — Origami gfx1101 support

Origami did not recognise navi32 at all: `arch_name_to_enum("gfx1101")` returned `Count`, so
every navi32 query fell through to a default. Added gfx1101 at **6 switch sites plus 3 table
entries**, copying gfx1100's values verbatim:

| file | site | what |
|---|---|---|
| `hardware.hpp` | `architecture_t` enum | new `gfx1101` member |
| `hardware.hpp` | `arch_name_to_enum` | `"gfx1101"` -> enum |
| `hardware.hpp` | `arch_enum_to_name` | enum -> `"gfx1101"` |
| `hardware.hpp` | `get_arch_constants` | gfx1100's `{7.12, 1.219*3.48, 0.732, 2, (0,0.11,0), 1.5}` |
| `hardware.hpp` | MI-latency map | WMMA V1 latencies (F16/BF16/I8 = 32, I4 = 16) |
| `hardware.cpp` | `cus_per_multiProcessorCount` | RDNA x2 list — **this is what makes 30 WGPs read as 60 CUs** |
| `hardware.cpp` | XCD count | 1 (monolithic) |
| `hardware.cpp` | 2 capability predicates | same answers as gfx1100 |

**Copying gfx1100's constants is correct rather than lazy for the CU count**, because the CU
count is *not* baked into them — it arrives from the device at runtime as
`multiProcessorCount * cus_per_multiProcessorCount`. A 60-CU part is therefore modelled
correctly without touching a single ratio. The constants that *would* need recalibration are
the memory ones (`mem2`/`mem3`): navi32 has less Infinity Cache (48-64 MB vs 64-96) and lower
bandwidth (432-624 GB/s vs 576-864). Noted inline in the source.

Verification: build is clean with no switch/enum warnings; `gfx1101` appears in the built
`libhipblaslt.so`; and the counts match — **6 `gfx1100` cases, 6 `gfx1101` cases**, so no
switch was missed. Origami's switches are exhaustive (they carry an explicit
`case architecture_t::Count`), so a missed site would have failed the build outright.

---

## P2 — Porting the navi31 catalog, gated on gfx1101

### The catalogs are disjoint, not nested

| | count |
|---|---|
| navi31 solutions | 298 |
| navi32 solutions | 73 |
| **shared by `SolutionNameMin`** | **0** |
| union | 371 |

navi32 is not a subset of navi31 — the two were tuned independently. Macro-tiles present in
navi31 and absent from navi32 include MT64x128x32 (21 kernels), MT128x64x32 (19),
MT64x64x32 (14), MT64x64x64 (14), MT128x96x32 (13).

**Their ProblemTypes also differ in 14 keys** (navi32's file predates fields such as
`SwizzleTensorA`, `DataTypeMXSA`, `MacDataTypeB`). Splicing the two solution lists into one
file would mix schemas, so the campaign compares **two complete, self-consistent logic files**
instead of merging them. That avoids the hazard entirely and keeps each arm buildable.

### GATE PASSED — all 298 navi31 solutions compile for gfx1101

`TensileCreateLibrary <logic> <out> HIP --architecture gfx1101` over navi31's catalog
retargeted to gfx1101:

| check | result |
|---|---|
| kernels processed | **298** |
| assembler errors | **0** |
| `overflowedResources` (VGPR / occupancy) | **0** |
| code object produced | `TensileLibrary_..._gfx1101.co` |
| ELF header | **`Flags: 0x46, gfx1101`** |

This is the user's requirement — *"if solution vgpr count is ok so it could compile to
navi32"* — enforced mechanically rather than assumed. It was worth testing rather than
inferring from "the caps are identical on paper": instruction selection can differ between
targets even at equal VGPR budgets. Here it does not, and now that is measured.

**Retargeting has two ISA sites**, and missing either yields a file that looks retargeted but
is not: top-level element `[2]`, and **every solution's own `ISA: [11,0,N]`**. The tool
(`retarget_logic.py`) rewrites both and hard-fails if any solution lacks the key.

---

## P3 — Measurement protocol: the CU mask had to be dropped from the sweep

The P0b CU mask gives genuinely correct 60-CU execution (62.4% vs an ideal 62.5%), but it is
**not usable for a long sweep**: `hipExtStreamCreateWithCUMask` streams intermittently wedge.
Measured on the same 8 shapes, same library, back to back:

| | completed | timeouts |
|---|---|---|
| without CU mask | 8 | **0** |
| with CU mask 60 | 5 | **3 (37%)** |

The hang is a teardown race, not a bad shape: the run emits its result row and then never
exits, and an isolated retry of the same (shape, arm) always passes. It is also
position-dependent rather than library-dependent — in the five-arm interleave, `navi32ship`
as the *first* arm had 0 timeouts while `navi32ship_aa`, the **same library** placed last,
had 2. At 35 s per hang and ~15% of 5 000 runs, that is hours of pure stall.

### Consequence: emulation is split, and each half is used where it is sound

| half | mechanism | used for |
|---|---|---|
| **selection** | `--sm_count_target 60` | the full 998-shape sweep — every arm picks kernels as a 60-CU part would |
| **execution** | `HIPBLASLT_BENCH_CU_MASK=60` | a smaller validation subset, to confirm the ranking survives real 60-CU timing |

**What the full sweep does and does not support.** Kernel *choice* is navi32-correct, and
because every arm runs on the same 96 CUs the execution error is common-mode — so **arm
ratios are meaningful**. Absolute throughput is optimistic for navi32 and must not be quoted
as a navi32 number. The residual risk is that a kernel which wins at 60 CUs may not show its
advantage at 96; that is exactly what the masked subset is for.

Unmasked, the sweep runs clean: **0 timeouts in the first 205 runs**, 0.79 runs/s.

### P3 interim (n=90 of 998) — WGM is a null; catalog depth is the whole story

| arm | geomean | wall-clock |
|---|---|---|
| A/A control (same library) | 99.74% | 99.97% |
| navi31 catalog @ WGM8 | **141.42%** | **124.04%** |
| navi31 catalog @ WGM6 | 141.26% | 123.07% |
| navi31 catalog @ WGM10 | 140.87% | 123.51% |

**The WGM hypothesis is not supported.** The three WGM variants sit within ~1 pt of each
other at every jackknife depth, against an A/A floor of ~0.3 pt. The tuning wiki names CU
count (and hence WGM) as *the* tuning-relevant difference between navi31 and navi32, and
WGM8 genuinely is ragged on 30 WGPs (3.75) — but re-forking it to 6 or 10, both clean factors
of 60 CUs, changes nothing measurable. A plausible mechanical story is not a result.

**What does matter is catalog depth**: navi31's 298 solutions beat navi32's 73 by **+41%
geomean / +24% wall-clock**.

And the win is not an outlier artefact — the jackknife runs the *other* way:

| dropped | A/A | wgm8 |
|---|---|---|
| 0 | 99.97% | 124.04% |
| 5 | 99.65% | 126.66% |
| 10 | 99.63% | 137.10% |
| 25 | 99.75% | 150.65% |
| 50 | 99.76% | 144.01% |

Removing the biggest time consumers *increases* the advantage, so the thin navi32 catalog is
hurting most on ordinary mid-sized shapes, not on a few giants. Concentration is high (top 5
shapes = 59% of kernel time), which is exactly why the jackknife is reported rather than a
single wall-clock number.

---

## P5 — Emitting a navi32 Prediction (Origami) library

A Prediction library has **no shape table** — element `[7]` is `None`, confirmed against the
shipped gfx942 StreamK Prediction files. Selection is Origami's analytical model evaluated
for the actual shape, over the solutions in `[5]`.

That is precisely what navi32 needs. Its GridBased table has **471 rows against navi31's
9 680**, so most real shapes are resolved to a distant neighbour. Replacing a sparse table
with a model evaluated per shape removes that failure mode — and Origami now knows gfx1101
(P1), so it can be evaluated for a 60-CU part.

Conversion is a small transformation (`to_prediction.py`): null out `[6]`–`[9]`, set
`[11] = Prediction`. Two arms built:

| arm | solutions | file |
|---|---|---|
| `pred298` | 298 (navi31 catalog) | 2 890 KB (was 3 318 KB with the table) |
| `pred73` | 73 (navi32 shipped) | 356 KB (was 386 KB) |

Verified the Prediction path is live: `pred298` loads and selects `MT128x128x64` where the
GridBased arm selected a different kernel for the same shape. Since a Prediction file has no
table to look up, that difference **is** the evidence Origami is doing the selecting.

### Where the thin catalog actually fails (n≈190, wall-clock vs shipped navi32)

| by size | n | ratio | | by geometry | n | ratio |
|---|---|---|---|---|---|---|
| large | 53 | 121% | | square | 67 | 121% |
| medium | 51 | 129% | | rect | 40 | 130% |
| small | 47 | 147% | | skinny | 48 | 132% |
| **tiny** | 20 | **231%** | | **gemv** | 16 | **270%** |

A/A control: 97.4–100.9% across every one of those cells.

**navi32's shipped catalog is not uniformly a bit thin — it collapses in a corner.** On GEMV
and tiny shapes navi31's catalog is 2.3–2.7x faster. That is the expected signature of a
471-row nearest-neighbour table: large square shapes are close to something in the table, but
a skinny or tiny shape gets matched to a distant neighbour and runs a badly-sized tile.

It also explains why the jackknife strengthens the result — the biggest time consumers are
large square shapes, which is precisely where the thin catalog does *least* badly.

### P3 FINAL (n=230) — WGM null confirmed, catalog depth confirmed

| arm | geomean | wall-clock |
|---|---|---|
| A/A control | 99.92% | 99.72% |
| navi31 catalog @ **WGM8** | **139.50%** | **124.50%** |
| navi31 catalog @ WGM6 | 139.01% | 124.44% |
| navi31 catalog @ WGM10 | 138.70% | 124.77% |

**WGM is a null: the three variants span 0.8 pt on geomean and 0.33 pt on wall-clock**,
against an A/A floor of 0.1–0.3 pt. Re-forking WorkGroupMapping to a clean factor of 60 CUs
changes nothing measurable.

That is worth stating plainly because it was the campaign's headline hypothesis, and it was
well-motivated: every shipped navi32 solution uses WGM8, 8 divides neither 30 WGPs (3.75) nor
60 CUs, and the tuning wiki names CU count as *the* tuning-relevant difference from navi31.
The mechanism was real; the effect is not. **A plausible mechanical story with a measurable
prediction is still only a hypothesis until measured** — and this one cost three of five arms
before the null was solid enough to stop.

**The real finding is catalog depth: +39.5% geomean / +24.5% wall-clock**, concentrated in
tiny and GEMV shapes (2.3–2.7x) where a 471-row nearest-neighbour table matches a distant
neighbour.

Data: `results/P3_wgm_final.csv` (1 150 rows, 5 arms, 0 timeouts).

---

## P6 — GridBased vs Origami-Prediction (n=124 of 998, stable across n=41/62/124)

| arm | geomean | wall-clock |
|---|---|---|
| A/A control | 99.71% | 99.66% |
| **`gridcat`** — navi31's 298 solutions, GridBased | **140.23%** | **123.71%** |
| `pred298` — the *same* 298 solutions, Origami-selected | 127.08% | 119.56% |
| `pred73` — navi32's 73 solutions, Origami-selected | 102.11% | 95.49% |

**The plan assumed the deliverable would be a Prediction (Origami) library. The measurement
says otherwise: keep GridBased.** Over an identical 298-solution pool, the dense
shape->solution table beats Origami's analytical selection by 13 pt geomean / 4 pt wall-clock.

Where Origami loses is specific — by size, wall-clock:

| | large | medium | small | tiny |
|---|---|---|---|---|
| `gridcat` | 121.8% | 121.4% | **149.3%** | 158.6% |
| `pred298` | 121.3% | 120.2% | **98.4%** | 155.9% |

Origami is level with GridBased on large, medium and tiny shapes and **collapses on small
ones** (98.4% vs 149.3%) — there it is no better than the thin shipped catalog it is supposed
to replace. That is consistent with prior work on this workspace, where no Origami
configuration reliably beat a shipped selector.

`pred73` is the clean control for "is Origami itself the problem, or the catalog?": Origami
over navi32's own 73 solutions lands at 102.1% geomean / 95.5% wall-clock — i.e. **swapping
selection alone buys nothing**. The gain is the catalog, not the selector.

### P6 at n=219 — unchanged across n = 41 / 62 / 124 / 219

| arm | geomean | wall-clock |
|---|---|---|
| A/A control | 99.76% | 99.75% |
| **`gridcat`** (298, GridBased) | **140.68%** | **124.62%** |
| `pred298` (298, Origami) | 126.41% | 120.68% |
| `pred73` (73, Origami) | 105.30% | 96.22% |

By size and geometry (wall-clock) — this is where the shipped catalog actually fails:

| | large | medium | small | tiny | | gemv | rect | skinny | square |
|---|---|---|---|---|---|---|---|---|---|
| `gridcat` | 121.1% | 126.5% | 144.9% | **229.5%** | | **260.0%** | 128.1% | 133.6% | 120.7% |
| `pred298` | 121.4% | 122.0% | 101.6% | 189.3% | | 206.8% | 123.0% | 120.5% | 118.5% |
| A/A | 100.0% | 98.6% | 99.8% | 99.5% | | 99.6% | 100.4% | 100.1% | 99.5% |

GEMV at **2.6x** and tiny at **2.3x** are the headline. A 471-row nearest-neighbour table
simply has nothing close to a GEMV shape, so it returns a tile sized for something else.

The sweep continues in the background toward all 998 shapes; it is resumable (the harness
skips `(shape, arm, rep)` triples already in the CSV) and the ranking has not moved across a
5x increase in n.

### P6 at n=601 — the geomean settles lower; wall-clock does not move

| arm | geomean | wall-clock |
|---|---|---|
| A/A control | 99.68% | 100.63% |
| **`gridcat`** | **129.29%** | **123.79%** |
| `pred298` | 118.15% | 120.65% |
| `pred73` | 99.85% | 92.58% |

**Correction to the earlier headline.** At n=219 the geomean read 140.7%; at n=601 it is
129.3%. The flops-weighted wall-clock barely moved (124.6% -> 123.8%). The small-n geomean
was optimistic — early shapes happened to over-represent the tiny/GEMV strata where the thin
catalog collapses, and geomean weights every shape equally, so those dominated. Wall-clock
weights by time and was insensitive to the same drift.

**The settled figures are +29% geomean / +24% wall-clock.** The commit message for the HHS
catalog quotes +40% geomean, taken at n=219; that number is too high and is corrected here.
The wall-clock claim in it stands.

Concentration also improved as the sample grew — top 5 shapes fell from 59% of kernel time at
n=41 to **19.8%** at n=601 — so the wall-clock figure is now far better conditioned than when
it was first quoted. The jackknife is flat (122.7–130.4% at every depth).

This is a good argument for not quoting a headline before the sample is broad: **the metric
that moved was the one weighting all shapes equally, and it moved by 11 points.**

---

## P6 FINAL — 996 shapes, 4 990 measurements, ZERO failures

| arm | geomean | wall-clock |
|---|---|---|
| A/A control (same library) | 99.72% | 100.32% |
| **`gridcat`** — navi31's 298 solutions, GridBased | **127.21%** | **123.91%** |
| `pred298` — same 298 solutions, Origami-selected | 115.68% | 120.45% |
| `pred73` — navi32's 73 solutions, Origami-selected | 98.33% | 93.25% |

**Headline: widening the navi32 TN HHS catalog is worth +27% geomean / +24% wall-clock.**

Robust: the jackknife is flat (123.2–126.4% at every depth), concentration is healthy at full
n (top 5 = 15.0% of kernel time, down from 59% at n=41), and the A/A control sits at
99.8–100.3% throughout.

Where it pays, wall-clock:

| | large | medium | small | tiny | | gemv | rect | skinny | square |
|---|---|---|---|---|---|---|---|---|---|
| `gridcat` | 122.7% | 122.7% | 140.2% | 141.7% | | **222.9%** | 123.3% | 124.4% | 123.5% |
| `pred298` | 122.2% | 119.4% | 107.4% | 127.2% | | 182.1% | 120.1% | 117.4% | 120.7% |
| A/A | 100.9% | 99.2% | 100.1% | 99.4% | | 99.5% | 99.8% | 99.3% | 100.8% |

**GEMV at 2.2x** is the standout, and small/tiny at ~1.4x. Large and square shapes gain a
uniform ~1.23x. This is the signature of a 471-row nearest-neighbour table: a large square
shape finds a near neighbour, a GEMV does not.

### The two rejected hypotheses, at full n

**Origami-Prediction stays behind GridBased** over the identical 298-solution pool — 115.7%
vs 127.2% geomean — and the gap is worst exactly where the catalog matters most
(small: 107.4% vs 140.2%). **`pred73` is the control that isolates the cause**: Origami over
navi32's *own* 73 solutions lands at 98.3% geomean / 93.2% wall-clock, i.e. **swapping the
selector alone makes things slightly worse**. The win is the catalog, not the selector.

**WGM re-forking is a null** (P3, n=230): 6/8/10 within 0.33 pt of each other on wall-clock
against a 0.3 pt A/A floor.

### Final correction to the shipped commit message

The HHS catalog commit quotes **+40% geomean**, measured at n=219. At n=996 the geomean is
**+27%**. The wall-clock claim (+24%) is unchanged — it read 124.6% at n=219 and 123.9% at
n=996.

That divergence is the useful part: **the equally-weighted metric moved 13 points as the
sample broadened while the time-weighted one moved 0.7.** Early shapes over-represented the
tiny/GEMV strata where the thin catalog collapses; geomean gave those shapes the same weight
as a 5 ms GEMM, wall-clock did not. Quote the wall-clock figure.

---

## P7 — Bounding the memory-bandwidth gap (the downclock probe was impossible)

The plan called for downclocking memory toward navi32's ~624 GB/s to bound how much this
960 GB/s card overstates memory-bound shapes. **That probe cannot be run here**: the system
has no clock control — `rocm-smi --showmclkrange` returns `get_od_volt, Not supported`,
`rocm-smi -s` reports every clock domain as "exists but EMPTY! Likely driver error", and
`pp_dpm_mclk` is absent from sysfs.

**Substitute that bounds the same gap.** Split the result by arithmetic intensity,
`AI = 2MNK / (2(MK + KN + 2MN))` flop/byte. High-AI shapes are compute-bound and insensitive
to bandwidth, so their measured ratio transfers to navi32 directly; low-AI shapes are where
the overstatement lives. The roofline crossover is ~125 flop/byte on this card and ~118 on
navi32, so the bands mean the same thing on both.

| band | n | % of kernel time | `gridcat` | A/A |
|---|---|---|---|---|
| memory-bound `AI<32` | 487 | **7.3%** | 142.89% | 101.67% |
| mixed `32–128` | 197 | 6.3% | 138.05% | 101.42% |
| compute-bound `128–512` | 195 | 13.6% | 124.12% | 99.34% |
| **deep compute `AI>=512`** | 117 | **72.8%** | **121.18%** | 100.28% |

**The caveat is much weaker than assumed. 73% of kernel time is deep-compute, and the win
there is 121% — essentially the headline figure.** The bands this card cannot speak for
(AI<32) are only 7.3% of kernel time, and they show the *largest* win (142.9%), so the
bandwidth difference is not propping up the result; if anything a more bandwidth-starved
navi32 would benefit more from a catalog that picks better tiles.

So the honest statement changes from "absolute throughput is optimistic, treat all numbers as
an upper bound" to: **the +24% wall-clock result is carried by compute-bound shapes and
transfers to navi32; only the 7.3% of time spent in memory-bound shapes remains unverifiable
on this hardware.**

Note the count/time inversion: memory-bound shapes are **49% of the shape count but 7.3% of
kernel time**. Reporting this split by shape count instead would have made the untestable
region look like half the suite.

---

## P4 — Catalog extension: closed on evidence, not run

The plan's next phase was to extend the catalog beyond navi31's 298 solutions and distil.
An oracle over what was already built shows the expected value is small:

| selector | wall-clock vs shipped | vs `gridcat` |
|---|---|---|
| shipped navi32 (73, GridBased) | 100.00% | 80.70% |
| **`gridcat` (298, GridBased)** | **123.91%** | 100.00% |
| ORACLE best-of-4-arms | 127.43% | **+2.84%** |
| ORACLE best of `gridcat`/`pred298` | 126.90% | +2.41% |

**A perfect per-shape selector over every arm built in this campaign beats `gridcat` by only
2.8%.** Going 73 -> 298 solutions captured +24%; perfect selection over the resulting pool
would add 2.8% more. `gridcat` is already the best of the four arms on **542 of 996 shapes
(54.4%)**.

So the pool is no longer the binding constraint, and neither is selection within it. Spending
hours adding kernels to a pool whose oracle is nearly exhausted is poor value against a 20 h
budget — better to stop with a measured, shipped +24% than to chase a bounded 2.8%.

**What this does NOT say.** The oracle spans two pools (73 and 298) under two selectors. It
bounds *recombining what exists*; it says nothing about what a catalog tuned natively for
60 CUs could achieve, since every kernel here was tuned for navi31's 96. That remains the
real open question, and it needs a tuning campaign on navi32 hardware — not more porting.

---

## Verification

Every headline figure recomputed from the raw CSV rather than quoted from prose:

| claim | report | recomputed |
|---|---|---|
| `gridcat` geomean | 127.21% | **127.21%** |
| `gridcat` wall-clock | 123.91% | **123.91%** |
| `pred298` geomean | 115.68% | **115.68%** |
| `pred73` wall-clock | 93.25% | **93.25%** |
| A/A geomean | 99.72% | **99.72%** |
| A/A wall-clock | 100.32% | **100.32%** |

**998 shapes x 5 arms = 4 990 rows exactly**, no duplicates, no missing, zero failures.

**Why the analysis uses 996 shapes, not 998.** Two shapes are degenerate — `M=4,N=1,K=8`
(32 flops) and `M=1,N=4,K=1` (4 flops). They ran fine (`status=ok`, ~14.7 us on every arm)
but GFLOPS underflows to `0.00`, so no ratio is definable and the analyzer drops them. All
five arms land within 0.8 us of each other on both, which is what you would expect from
measurements dominated entirely by kernel-launch overhead — they carry no information about
catalog quality either way.

Reproduce: `python3 analyze.py results/P6_main.csv`,
`python3 arith_intensity.py results/P6_main.csv`.

---

## P9 — bf16 (BBS) validation: the analogy holds across a dtype boundary

Three of the four catalogs shipped on analogy rather than measurement. The weakest link was
the **bf16 (BBS)** pair, because it crosses a dtype boundary from the measured fp16 case.
Measured it: 904 shapes, 3 arms (shipped 64-solution BBS, widened 306-solution BBS, and an
A/A control), bf16 in/out with fp32 compute.

| | fp16 HHS (measured earlier) | **bf16 BBS** |
|---|---|---|
| wall-clock | 123.91% | **122.17%** |
| geomean | 127.21% | 119.89% |
| A/A control | 100.32% | **100.60%** |

And the same shape-dependent signature:

| | large | medium | small | tiny | | gemv | rect | skinny | square |
|---|---|---|---|---|---|---|---|---|---|
| bf16 `bbs_wide` | 120.4% | 122.2% | 142.6% | 143.1% | | **178.1%** | 120.6% | 127.2% | 122.2% |
| fp16 `gridcat` | 122.7% | 122.7% | 140.2% | 141.7% | | 222.9% | 123.3% | 124.4% | 123.5% |

**+22% wall-clock on bf16 against +24% on fp16** (final n=996, 2 994 measurements,
zero failures), with the same ordering across every size and
geometry band. The mechanism is the catalog, not the dtype, exactly as the "same mechanism
applies" argument in the commit claimed — and now that claim rests on measurement across two
dtypes rather than on one measurement plus an analogy.

Two of the four shipped catalogs are now directly measured (HHS-TN and BBS-TN). The remaining
two are the `AuxH`/`AuxB` variants of those same two ProblemTypes — the closest possible
neighbours, differing only in the bias-aux epilogue.

### bf16 arithmetic-intensity bands — transfers at least as well as fp16

| band | n | % of kernel time | `bbs_wide` |
|---|---|---|---|
| memory-bound `AI<32` | 487 | 7.0% | 110.94% |
| mixed `32–128` | 197 | 6.0% | 114.15% |
| compute-bound `128–512` | 195 | 14.4% | 124.76% |
| **deep compute `AI>=512`** | 117 | **72.6%** | **123.60%** |

Same structure as fp16 — 72.6% of kernel time is deep-compute, winning 123.6%, essentially
the headline. Notably the bf16 win is **more** concentrated in the compute-bound bands
(110.9% in the memory-bound band vs fp16's 142.9%), so it depends even less on this card's
bandwidth advantage and transfers to navi32 with a smaller caveat than the fp16 result.

---

## Final state

| | |
|---|---|
| branch | `vmijovic/navi32`, 8 commits, pushed |
| measurements | **~9 100** across 3 sweeps, **zero failures** |
| shipped | 4 TN catalogs widened; Origami gains gfx1101; bench gains a CU mask |
| measured | HHS-TN **+24%**, BBS-TN **+22%** wall-clock |
| gated | all 935+298 solutions build for gfx1101 (`Flags: 0x46`) |
| rejected | WGM re-fork, Origami-Prediction, catalog extension — each with evidence |
| bounded | 73% of kernel time is compute-bound, so results transfer despite bandwidth |
| verified | every headline figure recomputed from raw CSV; all match |

---

## P10 — AuxH validation: three of four shipped catalogs now measured

| catalog | geomean | wall-clock | A/A |
|---|---|---|---|
| HHS-TN (fp16) | 127.21% | **123.91%** | 100.32% |
| BBS-TN (bf16) | 119.89% | **122.17%** | 100.60% |
| **AuxH-TN (fp16 + aux epilogue)** | 117.28% | **120.42%** | 100.42% |

All three complete: 996/996/997 shapes, zero failures. All land in a
**+20 to +24% wall-clock band**, with the same shape signature:

| AuxH by | large | medium | small | tiny | | gemv | rect | skinny | square |
|---|---|---|---|---|---|---|---|---|---|
| wall-clock | 119.5% | 119.2% | 137.1% | 146.0% | | **237.3%** | 119.0% | 122.7% | 120.2% |

GEMV at 237% is the largest of the three (HHS 223%, BBS 178%). The bias-aux epilogue does not
change the picture — as expected, since the epilogue is not what a sparse shape table gets
wrong.

**Only `AuxB` (bf16 + aux) remains unmeasured**, and it is now bracketed on both axes: the
same dtype as a measured case (BBS) and the same epilogue as a measured case (AuxH).

### A silent failure worth recording

The first AuxH attempt ran to completion at the expected rate and produced **231 rows that
were all `status=error`**, `gflops=0.00`, empty kernel name. The Aux ProblemTypes carry an
auxiliary output and need `--use_e --aux_type f16_r --activation_type gelu`; without them the
library reports `NO solution found`. (`--use_e` alone fails with *"activation type 1 does not
support '--use_e'"*.)

That is the third instance today of **a check returning the reassuring signal without doing
the work** — alongside the `--logic-filter` glob that built zero kernels while exiting 0, and
`ROC_GLOBAL_CU_MASK` reporting 30 CUs while restricting nothing. The rule now in the runbook:
**check status counts, not row counts.**

---

## Campaign totals

| | |
|---|---|
| branch | `vmijovic/navi32`, 10 commits, pushed |
| measurements | **15 122** across 5 completed sweeps, **zero failures in any** |
| shipped | 4 TN catalogs widened; Origami gains gfx1101; bench gains a CU mask |
| measured | **all four**: HHS +23.9%, BBS +22.2%, AuxH +20.4%, AuxB +18.8% wall-clock (996/996/997/997 shapes) |
| unmeasured | **none** — nothing on the branch rests on analogy |
| gated | every shipped solution builds for gfx1101, `Flags: 0x46` |
| rejected | WGM re-fork, Origami-Prediction, catalog extension — each with evidence |
| bounded | ~73% of kernel time is compute-bound, so results transfer despite bandwidth |
| wiki | gfx1101 page moved from "tentative, no data" to measured |

### The three silent failures

Each returned the reassuring signal without doing the work, and each would have produced a
confident wrong claim:

| what looked fine | what was actually happening |
|---|---|
| `ROC_GLOBAL_CU_MASK` reported 30 CUs | restricted nothing; each mask bit is a WGP, not a CU |
| `invoke build --logic-filter 'navi32/*'` exited 0 | compiled **zero** kernels; the glob does not recurse |
| AuxH sweep produced rows at the normal rate | all 231 were `status=error`, `gflops=0.00` |

The common rule, now in the runbook: **verify the artifact, not the exit status** — kernel
counts, ELF flags, throughput slopes, status counts. And where a knob appears inert, sweep it
before concluding: an inert knob and a mis-scaled one are indistinguishable at one point.

---

## P11 — AuxB measured: all four shipped catalogs now validated

| catalog | solutions | geomean | wall-clock | A/A |
|---|---|---|---|---|
| HHS-TN (fp16) | 73 -> 298 | 127.21% | **123.91%** | 100.32% |
| BBS-TN (bf16) | 64 -> 306 | 119.89% | **122.17%** | 100.60% |
| AuxH-TN (fp16 + aux) | 73 -> 313 | 117.28% | **120.42%** | 100.42% |
| **AuxB-TN (bf16 + aux)** | 64 -> 316 | 118.32% | **118.77%** | 100.19% |

**Nothing on the branch now rests on analogy.** All four land in a **+18.8% to +23.9%
wall-clock band**, every A/A control within 0.6 pt of 100%, and every one shows the same
signature — modest gains on large/square, large gains on small/tiny/GEMV:

| AuxB by | large | medium | small | tiny | | gemv | rect | skinny | square |
|---|---|---|---|---|---|---|---|---|---|
| wall-clock | 117.5% | 117.1% | 141.7% | 143.2% | | **174.6%** | 118.0% | 124.4% | 118.1% |

The ordering across all four is stable: **fp16 > bf16, non-aux > aux**, spanning 5 pt in
total. Neither dtype nor epilogue changes the mechanism — a 471-row nearest-neighbour table
mis-serves small and skinny shapes regardless of what is computed in them.

That the four agree this closely is the strongest evidence that the diagnosis is right. Had
the win come from something incidental to one ProblemType, four independent 1 000-shape
sweeps across two dtypes and two epilogues would not land within 5 pt of each other.

### Final audit — every figure recomputed from raw CSVs

| catalog | shapes | rows | geomean | wall-clock | A/A wall-clock | failures |
|---|---|---|---|---|---|---|
| HHS-TN (fp16) | 996 | 4 990 | 127.21% | **123.91%** | 100.32% | **0** |
| BBS-TN (bf16) | 996 | 2 994 | 119.89% | **122.17%** | 100.60% | **0** |
| AuxH-TN (fp16+aux) | 997 | 2 994 | 117.28% | **120.42%** | 100.42% | **0** |
| AuxB-TN (bf16+aux) | 997 | 2 994 | 118.32% | **118.77%** | 100.19% | **0** |
| WGM sweep (P3) | 230 | 1 150 | — | — | 99.72% | **0** |

**15 122 measurements, zero failures anywhere.** Every published number recomputed from the
CSVs rather than quoted from prose, and all match.
