# navi33 (gfx1102): evidence for a decision, not a change

**Nothing here is shipped.** No navi33 logic file is modified on this branch. This is the
evidence needed to decide whether a navi33 campaign is worth authorising, gathered because that
question cannot be answered without it.

## 1. navi33 ships a byte-identical copy of navi32's thin catalog

Earlier notes recorded the cross-architecture gap as "same counts, same profile". It is
stronger than that:

| | navi31 | navi33 |
|---|---|---|
| solutions (TN HHS) | **298** | **73** |
| logic file lines | 114 299 | 14 674 |
| shape-table rows | 9 680 | **471** |

Built both navi32's *shipped* catalog and navi33's for gfx1100 and compared the code objects:
**60 kernel symbols each, identical sets, 0 unique to either side**, and 73 solutions each.

**Confirmed at the source level, and it is five architectures, not two.** Comparing the logic
files directly (`navi32` taken from `origin/develop`, i.e. before this branch widened it):

| | HHS | BBS | AuxH | AuxB |
|---|---|---|---|---|
| **navi32 (pre-fix), navi33, gfx1103, gfx1150, gfx1152** | 14 674 | 12 991 | 14 674 | 12 991 |
| navi31 | 114 299 | 117 127 | 118 846 | 119 653 |
| gfx1151 *(the one arch with a real tuning campaign)* | 131 721 | 131 460 | 129 297 | 129 297 |
| gfx1153 *(its own variant)* | 20 865 | 12 991 | 14 674 | 12 991 |

Those five files are **identical**, for **all four** thin ProblemTypes. A raw `diff` of any pair
is **152 lines out of 14 674, containing only 7 distinct contents**, every one an identifier:

```
- [Device 73f0]   vs  - [Device 150e]     # PCI device ID
- navi33          vs  - gfx1150           # arch name, element [1]
  ISA: [11, 0, 2] vs    ISA: [11, 5, 0]   # x73, one per solution
```

Nothing else differs — not one solution parameter, not one of the 471 table rows. The string
`gfx1102` occurs exactly **once** in navi33's file (the arch-name line), so kernel names are not
arch-tagged and this is not an artefact of over-normalising.

**So the defect is one artefact copied five times, not five architectures independently
under-tuned.** gfx1151 is the counter-example that proves the point: it is the only RDNA3 part
that got its own tuning campaign, and it is the only one with a full-size catalog. This branch
fixes one of the five.

## 2. At navi33's 32 CUs, widening is worth ~+15% wall-clock

`n33ship` (73 solutions) vs `wide` (navi31's 298), 205 shapes over all 109 strata, genuine
32-CU execution (16 of 48 WGPs masked), `--sm_count_target 32`, 2 reps:

| arm | geomean | wall-clock |
|---|---|---|
| `n33ship_aa` (A/A control) | 99.85% | **100.00%** |
| `wide` | **125.22%** | **115.31%** |

**A/A floor of 0.00 pt** — the cleanest control in the campaign.

Jackknife is strongly favourable: dropping the largest time consumers **raises** the win
(115.3 -> 121.7 -> 128.0 -> 135.5 after 5/10/25/50), so it is not carried by a few big shapes.

| by size | wall-clock | | by geometry | wall-clock |
|---|---|---|---|---|
| large (37) | 111.4% | | **gemv (11)** | **263.6%** |
| medium (67) | 122.7% | | skinny (63) | 131.4% |
| small (52) | 135.1% | | rect (66) | 118.7% |
| tiny (49) | 134.0% | | square (65) | 111.7% |

## 3. The gain is smaller at 32 CUs than at 60, and the reason is mechanical

Same catalogs, same shapes, only the CU count differs:

| | geomean | wall-clock |
|---|---|---|
| 60 CUs (navi32) | 125.66% | **122.74%** |
| 32 CUs (navi33) | 125.37% | **114.91%** |

**Per-shape the catalog helps identically** (geomean 125.4 vs 125.7). The wall-clock difference
is a *weighting* effect: large shapes gain least (111.4% at 32 CUs vs 121.4% at 60), and at
fewer CUs they occupy proportionally more of the total time, so they pull the time-weighted
metric down. Nothing about the catalog changed; the mix did.

> *Method note.* `analyze.py` reported geomean **125.22%** for **both** runs — identical to two
> decimals across different hardware configurations, which is the kind of too-neat number worth
> distrusting. Recomputed independently it is 125.66 / 125.37; `analyze.py` aggregates reps
> differently and both happened to round together. Coincidence, not a bug — but checked rather
> than published.

## 4. What this does and does not establish

**Established:**
- navi33's catalog is the same thin artefact navi32 shipped (identical kernels).
- Widening it is worth **~+15% wall-clock / +25% geomean** at navi33's CU count, against a
  0.00 pt A/A floor.
- **Buildability is not a blocker**: navi31's four TN catalogs retargeted to gfx1102 build
  987 kernels, 0 assembler errors, 0 `overflowedResources`, ELF `Flags: 0x47, gfx1102`.
- **Occupancy is not the blocker it looked like**: only **10.1%** of kernel time sits in
  solutions that lose waves on navi33's smaller register file, because LDS binds 88% of these
  kernels ([`NAVI33_OCCUPANCY.md`](NAVI33_OCCUPANCY.md)).

**Not established — the honest gaps:**
- Kernels were **built for gfx1100 and executed on gfx1100**. The CU count is emulated
  faithfully (verified: 32-CU/60-CU throughput ratio 0.551 against an ideal 0.533); the
  **register file and memory system are not**. gfx1102 buildability is established separately,
  but no gfx1102 *code* was executed.
- navi33's memory system differs and is not emulable here, the same caveat that applies to the
  navi32 result.
- Only **TN HHS** was measured. navi33 has the same four thin ProblemTypes; the other three are
  inferred from navi32, where all four behaved alike.

## Reproduce

```bash
# retarget navi33's shipped catalog to gfx1100 and build it
python3 retarget_logic.py <navi33 TN HHS yaml> arms/navi33ship/x.yaml --isa gfx1100 --name navi31
./build_arm.sh navi33ship
python3 bench_arms.py \
  --arms n33ship=$HOME/navi32/libs/navi33ship/library/gfx1100 \
         wide=$HOME/navi32/libs/wgm8/library/gfx1100 \
         n33ship_aa=$HOME/navi32/libs/navi33ship/library/gfx1100 \
  --shapes state/eval_shapes_masked.json --out results/P18_navi33_32cu.csv \
  --reps 2 --cus 32 --fixed-iters 20 --timeout 45
python3 analyze.py results/P18_navi33_32cu.csv n33ship
```

**Verify the CU mask by throughput, never by a reported count** — compare a shape measured at
two mask sizes and check the ratio tracks the CU ratio. Here 32/60 measured 0.551 against an
ideal 0.533.
