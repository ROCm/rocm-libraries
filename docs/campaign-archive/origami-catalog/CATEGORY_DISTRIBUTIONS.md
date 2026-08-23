# Per-category distributions — by catalog and selector

![distributions](category_distributions.png)

Each panel is a cumulative curve: x is a shape's ratio to the shipped selector
(G0), y is the fraction of shapes at or below it. **Left of the dashed line is
slower than G0.** A steep curve means consistency; a long left foot means a tail
of bad shapes that a geomean hides. The figure plots four arms for legibility;
the tables below cover all of them.

## Overall — all 1,500 shapes

| catalog | kernels | selector | ALL | worst 10% | best 10% |
|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | **100.00** | 100.00 | 100.00 |
| Grid pool | 298 | tuned Origami | 95.51 | 65.31 | 124.27 |
| Grid pool | 298 | stock Origami | 92.96 | 61.31 | 123.32 |
| SK3 v1 | 192 | tuned Origami | 96.85 | 64.40 | 138.66 |
| SK3 v1 | 192 | stock Origami | 96.92 | 64.88 | 139.17 |
| v2 union | 104 | tuned Origami | 94.89 | 63.98 | 127.21 |
| v2 union | 104 | stock Origami | 94.85 | 63.67 | 127.88 |
| v3 guard | 76 | tuned Origami | 98.46 | 61.69 | 146.28 |
| v3 guard | 76 | stock Origami | 97.70 | 60.41 | 146.22 |
| v4 3-bucket | 82 | tuned Origami | 96.46 | 62.99 | 139.32 |
| v5 traps | 61 | tuned Origami | 99.92 | 62.10 | 156.19 |
| v5 traps | 61 | stock Origami | 98.81 | 58.97 | 158.42 |
| v6 global | 58 | tuned Origami | 100.05 | 62.10 | 159.92 |
| v6 global | 58 | stock Origami | 99.59 | 59.66 | 157.39 |
| v7 time | 45 | tuned Origami | 100.58 | 62.69 | 165.21 |
| v7 time | 45 | stock Origami | 99.15 | 58.81 | 163.68 |
| hybrid_slim | 58+120 | both, size-gated | 100.66 | 83.49 | 117.95 |

Tails are **weighted by size category**: each category's own worst/best tenth,
combined in proportion to how many shapes that category holds. Pooling all 1,500
shapes instead would not give a global number — the pooled bottom decile is ~86%
tiny+small for the Origami arms (49% of the set) and 0–5% large, but ~80%
medium+large for `hybrid_slim`, so the two would not be the same statistic.

These are the columns to select on when the artifact will serve unknown shapes:
a catalog can carry a healthy mean while a tenth of shapes in every regime run
far below the baseline.

## By size — `geomean [worst 10% / best 10%]`, % of G0

| catalog | kernels | selector | tiny | small | medium | large |
|---|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] |
| Grid pool | 298 | tuned Origami | 94.9  [60 / 125] | 92.1  [53 / 129] | 97.3  [70 / 126] | 98.2  [88 / 111] |
| Grid pool | 298 | stock Origami | 92.6  [53 / 126] | 89.3  [49 / 129] | 94.2  [68 / 124] | 96.9  [87 / 110] |
| SK3 v1 | 192 | tuned Origami | 110.2  [67 / 206] | 89.7  [53 / 131] | 95.8  [66 / 126] | 94.0  [78 / 109] |
| SK3 v1 | 192 | stock Origami | 110.8  [68 / 204] | 91.6  [54 / 132] | 93.7  [66 / 126] | 94.7  [77 / 111] |
| v2 union | 104 | tuned Origami | 92.1  [57 / 129] | 91.3  [52 / 132] | 97.8  [70 / 127] | 98.6  [86 / 118] |
| v2 union | 104 | stock Origami | 93.3  [56 / 131] | 90.4  [50 / 133] | 97.1  [72 / 127] | 99.6  [88 / 117] |
| v3 guard | 76 | tuned Origami | 100.3  [53 / 188] | 89.4  [52 / 131] | 102.5  [69 / 142] | 103.1  [79 / 133] |
| v3 guard | 76 | stock Origami | 105.0  [54 / 204] | 88.8  [48 / 132] | 99.1  [68 / 137] | 100.2  [78 / 126] |
| v4 3-bucket | 82 | tuned Origami | 96.8  [54 / 176] | 90.0  [51 / 131] | 99.4  [70 / 133] | 100.8  [86 / 123] |
| v5 traps | 61 | tuned Origami | 100.3  [54 / 189] | 90.5  [49 / 139] | 105.4  [73 / 159] | 104.5  [79 / 141] |
| v5 traps | 61 | stock Origami | 105.5  [52 / 209] | 87.2  [44 / 137] | 102.1  [69 / 158] | 103.0  [78 / 140] |
| v6 global | 58 | tuned Origami | 100.6  [54 / 207] | 90.0  [49 / 139] | 106.1  [73 / 160] | 104.4  [79 / 143] |
| v6 global | 58 | stock Origami | 104.8  [51 / 204] | 87.3  [45 / 138] | 104.0  [71 / 156] | 105.0  [79 / 139] |
| v7 time | 45 | tuned Origami | 101.1  [54 / 209] | 89.3  [47 / 138] | 107.4  [75 / 166] | 105.6  [83 / 158] |
| v7 time | 45 | stock Origami | 105.0  [49 / 209] | 86.6  [44 / 139] | 103.7  [72 / 165] | 103.8  [78 / 152] |
| hybrid_slim | 58+120 | both, size-gated | 103.8  [98 / 111] | 102.7  [84 / 122] | 99.7  [77 / 123] | 95.5  [78 / 111] |

n: tiny 337, small 399, medium 504, large 260. Deciles are geomeans over the bottom and top tenth, so one outlier cannot
define them.

## By geometry — `geomean [worst 10% / best 10%]`, % of G0

| catalog | kernels | selector | gemv | skinny | rect | square |
|---|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] |
| Grid pool | 298 | tuned Origami | 93.9  [60 / 131] | 94.3  [56 / 136] | 96.2  [67 / 121] | 96.2  [70 / 115] |
| Grid pool | 298 | stock Origami | 93.4  [56 / 132] | 92.2  [55 / 135] | 92.9  [58 / 120] | 93.7  [62 / 114] |
| SK3 v1 | 192 | tuned Origami | 108.4  [62 / 212] | 94.1  [58 / 138] | 97.9  [67 / 148] | 96.4  [65 / 143] |
| SK3 v1 | 192 | stock Origami | 107.7  [62 / 205] | 93.3  [57 / 138] | 98.4  [69 / 150] | 97.0  [67 / 148] |
| v2 union | 104 | tuned Origami | 92.0  [57 / 129] | 93.9  [56 / 135] | 95.2  [63 / 126] | 96.1  [68 / 120] |
| v2 union | 104 | stock Origami | 93.9  [59 / 133] | 94.5  [59 / 138] | 95.2  [62 / 125] | 95.0  [61 / 119] |
| v3 guard | 76 | tuned Origami | 100.6  [58 / 210] | 94.7  [52 / 147] | 99.4  [63 / 143] | 100.9  [67 / 144] |
| v3 guard | 76 | stock Origami | 104.9  [58 / 204] | 95.6  [55 / 147] | 97.6  [59 / 148] | 98.5  [60 / 150] |
| v4 3-bucket | 82 | tuned Origami | 95.9  [53 / 187] | 93.3  [53 / 141] | 97.4  [64 / 137] | 98.8  [69 / 135] |
| v5 traps | 61 | tuned Origami | 99.2  [58 / 208] | 97.2  [54 / 161] | 100.5  [60 / 152] | 102.1  [67 / 147] |
| v5 traps | 61 | stock Origami | 105.0  [58 / 204] | 96.8  [53 / 164] | 98.5  [56 / 156] | 100.0  [57 / 161] |
| v6 global | 58 | tuned Origami | 98.4  [58 / 209] | 97.2  [54 / 161] | 100.8  [60 / 158] | 102.4  [65 / 159] |
| v6 global | 58 | stock Origami | 106.4  [59 / 208] | 97.8  [55 / 164] | 99.1  [55 / 155] | 100.6  [56 / 159] |
| v7 time | 45 | tuned Origami | 98.8  [58 / 213] | 97.7  [54 / 169] | 101.6  [61 / 167] | 102.7  [63 / 165] |
| v7 time | 45 | stock Origami | 106.0  [58 / 210] | 97.6  [54 / 170] | 98.7  [54 / 163] | 99.9  [54 / 166] |
| hybrid_slim | 58+120 | both, size-gated | 103.5  [98 / 111] | 102.7  [82 / 128] | 100.1  [82 / 116] | 98.8  [79 / 114] |

n: gemv 87, skinny 452, rect 496, square 465.

> **Treat the `gemv` deciles with care.** At n=87 each decile is a mean over 8
> shapes, so it moves a lot run to run — the campaign's noise floor is ~1 point
> on 1,500 shapes and correspondingly worse here. The other three columns
> (452/496/465) are on the same footing as the size table. The curves in the
> figure are the more reliable read for `gemv`.

Tiered iteration protocol; ratios against the G0 row are protocol-dependent by
~5 points (see `FINAL_CATALOG_REPORT.md` §0a). Comparisons *between* the other
rows are not.
