# WP-L2c tables (ratios only)

Companion of `WP-L2c.md`. Absolute times: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2c.md`.

## 1. Final builds, both nodes (jobs 68241619 n01-09 bus 65 / 68241620 n07-17 bus 65; 24 rounds; value pairs n01-09 / n07-17)

Arm time over the final chain's (median; W / L = the win rule against the chain's second graph and recompiled copy,
from the arm's side: L = the arm is slower). `first base` = the climb's starting build; `other-state base` = the other
state's final build on its own data; `S3 / S4 / S5 alone` = the chain with only that stage's work (all edges kept);
`edges` = no stage work. Conv epilogue price = standalone E3 / plain E1 on the same plan (streamed); hand-off price =
E4 / P-L1's kernel (streamed).

| state | TP | rows | first base / final | other-state base / this | per-stage launches / chain | one launch per layer / chain | S3 alone / chain | S4 alone / chain | S5 alone / chain | edges / chain | conv epilogue price (E3 / E1) | hand-off price (E4 / P-L1 kernel) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 5 | 1.204 L / 1.193 L | 0.947 W / 0.966 W | 1.371 L / 1.376 L | 1.291 L / 1.290 L | 0.571 W / 0.554 W | 0.450 W / 0.438 W | 0.391 W / 0.382 W | 0.167 W / 0.165 W | 1.08 / 1.09 | 1.022 / 1.019 |
| f32 | 1 | 8 | 1.156 L / 1.151 L | 0.905 W / 0.911 W | 1.348 L / 1.341 L | 1.234 L / 1.220 L | 0.504 W / 0.505 W | 0.486 W / 0.487 W | 0.357 W / 0.356 W | 0.146 W / 0.140 W | 1.12 / 1.14 | 1.021 / 1.003 |
| f32 | 1 | 16 | 1.159 L / 1.165 L | 0.895 W / 0.900 W | 1.248 L / 1.254 L | 1.187 L / 1.180 L | 0.436 W / 0.440 W | 0.582 W / 0.581 W | 0.304 W / 0.303 W | 0.125 W / 0.129 W | 1.18 / 1.19 | 1.005 / 1.004 |
| f32 | 1 | 32 | 1.215 L / 1.210 L | 0.819 W / 0.809 W | 1.185 L / 1.175 L | 1.117 L / 1.122 L | 0.349 W / 0.344 W | 0.704 W / 0.702 W | 0.220 W / 0.217 W | 0.124 W / 0.124 W | 1.10 / 1.12 | 1.005 / 0.999 |
| f32 | 1 | 64 | 1.228 L / 1.221 L | 0.720 W / 0.712 W | 1.113 L / 1.108 L | 1.067 L / 1.066 L | 0.264 W / 0.257 W | 0.744 W / 0.752 W | 0.171 W / 0.165 W | 0.085 W / 0.082 W | 1.55 / 1.55 | 1.011 / 1.010 |
| f32 | 1 | 128 | 1.116 L / 1.117 L | 0.682 W / 0.681 W | 1.063 L / 1.074 L | 1.038 L / 1.036 | 0.207 W / 0.202 W | 0.782 W / 0.797 W | 0.113 W / 0.109 W | 0.047 W / 0.046 W | 1.73 / 1.75 | 1.005 / 1.002 |
| f32 | 2 | 5 | 1.234 L / 1.236 L | 0.951 W / 0.953 W | 1.376 L / 1.376 L | 1.343 L / 1.346 L | 0.626 W / 0.622 W | 0.435 W / 0.425 W | 0.389 W / 0.374 W | 0.181 W / 0.168 W | 1.17 / 1.17 | 1.021 / 1.041 |
| f32 | 2 | 8 | 1.405 L / 1.401 L | 0.958 W / 0.950 W | 1.409 L / 1.402 L | 1.372 L / 1.372 L | 0.542 W / 0.543 W | 0.516 W / 0.508 W | 0.387 W / 0.394 W | 0.194 W / 0.201 W | 1.22 / 1.24 | 1.015 / 1.012 |
| f32 | 2 | 16 | 1.303 L / 1.305 L | 0.861 W / 0.857 W | 1.251 L / 1.247 L | 1.271 L / 1.264 L | 0.530 W / 0.523 W | 0.541 W / 0.534 W | 0.303 W / 0.305 W | 0.155 W / 0.158 W | 1.15 / 1.15 | 1.013 / 0.997 |
| f32 | 2 | 32 | 1.282 L / 1.282 L | 0.890 W / 0.886 W | 1.237 L / 1.239 L | 1.208 L / 1.213 L | 0.452 W / 0.447 W | 0.638 W / 0.638 W | 0.257 W / 0.258 W | 0.146 W / 0.150 W | 1.18 / 1.18 | 1.005 / 0.992 |
| f32 | 2 | 64 | 1.253 L / 1.242 L | 0.789 W / 0.789 W | 1.162 L / 1.159 L | 1.120 L / 1.111 L | 0.342 W / 0.339 W | 0.712 W / 0.719 W | 0.223 W / 0.224 W | 0.124 W / 0.129 W | 1.42 / 1.42 | 1.004 / 1.015 |
| f32 | 2 | 128 | 1.094 L / 1.093 L | 0.763 W / 0.755 W | 1.109 L / 1.107 L | 1.073 L / 1.068 L | 0.304 W / 0.297 W | 0.752 W / 0.754 W | 0.138 W / 0.134 W | 0.079 W / 0.080 W | 1.39 / 1.41 | 1.004 / 1.002 |
| bf16 | 1 | 5 | 1.201 L / 1.194 L | 1.051 L / 1.041 L | 1.370 L / 1.358 L | 1.289 L / 1.279 L | 0.588 W / 0.585 W | 0.396 W / 0.391 W | 0.397 W / 0.395 W | 0.165 W / 0.161 W | 1.08 / 1.09 | 1.019 / 1.012 |
| bf16 | 1 | 8 | 1.185 L / 1.171 L | 1.108 L / 1.104 L | 1.371 L / 1.362 L | 1.272 L / 1.263 L | 0.554 W / 0.549 W | 0.444 W / 0.434 W | 0.389 W / 0.389 W | 0.156 W / 0.150 W | 1.14 / 1.15 | 1.018 / 1.013 |
| bf16 | 1 | 16 | 1.169 L / 1.172 L | 1.168 L / 1.160 L | 1.284 L / 1.282 L | 1.215 L / 1.208 L | 0.505 W / 0.507 W | 0.476 W / 0.479 W | 0.327 W / 0.340 W | 0.135 W / 0.145 W | 1.20 / 1.20 | 1.009 / 1.000 |
| bf16 | 1 | 32 | 1.277 L / 1.288 L | 1.374 L / 1.382 L | 1.207 L / 1.220 L | 1.178 L / 1.180 L | 0.428 W / 0.433 W | 0.588 W / 0.587 W | 0.265 W / 0.262 W | 0.136 W / 0.131 W | 1.38 / 1.40 | 1.011 / 0.998 |
| bf16 | 1 | 64 | 1.311 L / 1.317 L | 1.465 L / 1.492 L | 1.138 L / 1.149 L | 1.097 L / 1.095 L | 0.367 W / 0.366 W | 0.623 W / 0.628 W | 0.228 W / 0.223 W | 0.108 W / 0.105 W | 1.59 / 1.61 | 1.002 / 1.009 |
| bf16 | 1 | 128 | 1.175 L / 1.164 L | 1.581 L / 1.593 L | 1.092 L / 1.082 L | 1.057 / 1.049 | 0.317 W / 0.312 W | 0.666 W / 0.666 W | 0.173 W / 0.169 W | 0.072 W / 0.073 W | 1.71 / 1.74 | 1.012 / 1.008 |
| bf16 | 2 | 5 | 1.273 L / 1.266 L | 1.036 L / 1.036 L | 1.399 L / 1.399 L | 1.355 L / 1.358 L | 0.649 W / 0.647 W | 0.388 W / 0.386 W | 0.403 W / 0.407 W | 0.170 W / 0.173 W | 1.17 / 1.17 | 1.013 / 1.016 |
| bf16 | 2 | 8 | 1.449 L / 1.437 L | 1.074 L / 1.072 L | 1.432 L / 1.427 L | 1.385 L / 1.369 L | 0.571 W / 0.566 W | 0.461 W / 0.454 W | 0.416 W / 0.405 W | 0.197 W / 0.199 W | 1.25 / 1.27 | 1.017 / 1.017 |
| bf16 | 2 | 16 | 1.430 L / 1.425 L | 1.223 L / 1.218 L | 1.309 L / 1.305 L | 1.325 L / 1.314 L | 0.526 W / 0.526 W | 0.505 W / 0.511 W | 0.359 W / 0.363 W | 0.168 W / 0.163 W | 1.20 / 1.21 | 1.018 / 1.007 |
| bf16 | 2 | 32 | 1.321 L / 1.334 L | 1.223 L / 1.222 L | 1.248 L / 1.253 L | 1.268 L / 1.271 L | 0.516 W / 0.515 W | 0.540 W / 0.534 W | 0.287 W / 0.284 W | 0.159 W / 0.157 W | 1.17 / 1.18 | 1.003 / 1.001 |
| bf16 | 2 | 64 | 1.280 L / 1.292 L | 1.367 L / 1.395 L | 1.198 L / 1.203 L | 1.184 L / 1.180 L | 0.412 W / 0.410 W | 0.589 W / 0.590 W | 0.270 W / 0.262 W | 0.133 W / 0.130 W | 1.50 / 1.50 | 1.004 / 1.025 |
| bf16 | 2 | 128 | 1.110 L / 1.105 L | 1.474 L / 1.486 L | 1.121 L / 1.123 L | 1.105 L / 1.097 L | 0.422 W / 0.419 W | 0.627 W / 0.627 W | 0.186 W / 0.184 W | 0.101 W / 0.099 W | 1.41 / 1.43 | 1.008 / 1.012 |

## 2. Final knobs per cell (`cfg/base_final.json`)

Plans: mt (16-row weight tiles per wave), kw (K split over wave groups), splits (K splits over CTAs), token tiles per unit,
X source, weight loads in flight, weight cache policy. All E1 stages run 8 waves with op_sel scales and tile weights.

| cell | E3 plan (mt, kw, splits, token tiles per unit, X source, depth, weight cache) | E5 plan | conv epilogue | S4 / S5 prefetch | LDS pad | recurrence depth |
|---|---|---|---|---|---|---|
| 1:5:f32 | 1, 1, 1, 1, global, 4, default | 1, 8, 1, 1, lds, 2, nt | batch + cpre | S4 state warm + S5 weights | fit | 1 |
| 1:8:f32 | 2, 2, 1, 1, global, 4, default | 1, 4, 1, 1, lds, 4, nt | lds | S4 state warm | fit | 1 |
| 1:16:f32 | 2, 2, 1, 1, lds, 4, default | 1, 4, 1, 1, lds, 4, nt | lds | S4 state warm | fit | 1 |
| 1:32:f32 | 2, 2, 1, 2, lds, 4, default | 1, 4, 1, 1, lds, 4, default | lds | S4 state warm | fit | 2 |
| 1:64:f32 | 1, 1, 1, 4, lds, 4, nt | 1, 4, 1, 2, lds, 4, default | batch (groups of 4) | S4 state warm | fit | 2 |
| 1:128:f32 | 2, 1, 1, 4, lds, 4, nt | 1, 2, 1, 2, lds, 6, default | batch (groups of 4) | S4 state warm (2 chunk steps) | fit | 2 |
| 2:5:f32 | 2, 2, 1, 1, lds, 4, default | 1, 8, 1, 1, global, 4, default | lds | S4 state warm | fit | 1 |
| 2:8:f32 | 2, 4, 1, 1, global, 4, default | 1, 8, 1, 1, lds, 4, nt | lds | S4 state warm | entry | 1 |
| 2:16:f32 | 1, 2, 1, 1, lds, 4, nt | 1, 8, 1, 1, lds, 4, nt | lds | S4 state warm | fit | 1 |
| 2:32:f32 | 1, 2, 1, 2, lds, 4, nt | 1, 4, 1, 1, lds, 4, default | batch (groups of 4) | S4 state warm + S5 weights | entry | 1 |
| 2:64:f32 | 1, 1, 1, 2, lds, 4, nt | 1, 4, 1, 2, lds, 3, default | batch | S4 state warm | fit | 2 |
| 2:128:f32 | 2, 2, 1, 4, lds, 4, default | 1, 2, 1, 2, lds, 6, default | lane + cpre | S4 state warm (2 chunk steps) | fit | 2 |
| 1:5:bf16 | 1, 1, 1, 1, global, 4, default | 1, 8, 1, 1, lds, 2, nt | batch + cpre | S4 state warm + S5 weights | fit | 1 |
| 1:8:bf16 | 2, 2, 1, 1, global, 4, default | 1, 4, 1, 1, lds, 4, nt | lds | S4 state warm | fit | 1 |
| 1:16:bf16 | 2, 2, 1, 1, lds, 4, default | 1, 4, 1, 1, lds, 4, nt | lds | S4 state warm | fit | 1 |
| 1:32:bf16 | 1, 1, 1, 2, lds, 4, nt | 1, 4, 1, 1, lds, 4, default | batch + cpre | S4 state warm + S5 weights | fit | 2 |
| 1:64:bf16 | 1, 1, 1, 4, lds, 4, nt | 1, 4, 1, 2, lds, 4, default | batch (groups of 4) | S4 state warm | fit | 2 |
| 1:128:bf16 | 2, 1, 1, 4, lds, 4, nt | 1, 2, 1, 2, lds, 6, default | batch (groups of 4) | S4 state warm (2 chunk steps) | fit | 2 |
| 2:5:bf16 | 2, 2, 1, 1, lds, 4, default | 1, 8, 1, 1, global, 4, default | lds | S4 state warm | fit | 1 |
| 2:8:bf16 | 2, 4, 1, 1, global, 4, default | 1, 8, 1, 1, lds, 4, nt | lds | S4 state warm | entry | 1 |
| 2:16:bf16 | 2, 4, 1, 1, global, 2, nt | 1, 8, 1, 1, lds, 4, nt | lds | S4 state warm | entry | 1 |
| 2:32:bf16 | 1, 2, 1, 2, lds, 4, nt | 1, 4, 1, 1, lds, 4, default | batch (groups of 4) + cpre | S4 state warm + S5 weights | entry | 1 |
| 2:64:bf16 | 1, 1, 1, 2, lds, 4, nt | 1, 4, 1, 2, lds, 3, default | batch | S4 state warm | fit | 2 |
| 2:128:bf16 | 2, 2, 1, 4, lds, 4, default | 1, 2, 1, 2, lds, 6, default | lane + cpre | S4 state warm (2 chunk steps) | fit | 2 |

## 3. Accepted changes of the climb (`cfg/ledger.jsonl`)

| round | cell | accepted change | median ratio n01-09 / n07-17 | verdicts |
|---|---|---|---|---|
| r1+r2 | 1:5:f32 | `e3c3+e5c3+cepi_fit+pre_s4` | 0.858 / 0.867 | WIN / WIN |
| r1+r2 | 1:8:f32 | `e3c4+e5f1+cepi+pre_s4` | 0.858 / 0.864 | WIN / WIN |
| r1+r2 | 1:16:f32 | `e3c6+e5f1+cepi_fit+pre_s4` | 0.851 / 0.860 | WIN / WIN |
| r1+r2 | 1:32:f32 | `e5c1+cepi+pre_s4+lpad` | 0.820 / 0.829 | WIN / WIN |
| r1+r2 | 1:64:f32 | `e5f6+cbatch_g4+pre_s4` | 0.875 / 0.876 | WIN / WIN |
| r1+r2 | 1:128:f32 | `e3f7+e5f14+cbatch_g4+pre_s4w2` | 0.941 / 0.939 | WIN / WIN |
| r1+r2 | 2:5:f32 | `e3c6+e5c3+cepi+pre_s4` | 0.809 / 0.805 | WIN / WIN |
| r1+r2 | 2:8:f32 | `e3c6+e5c0+cepi+pre_s4` | 0.789 / 0.791 | WIN / WIN |
| r1+r2 | 2:16:f32 | `e3f4+e5c0+cepi+pre_s4` | 0.767 / 0.780 | WIN / WIN |
| r1+r2 | 2:32:f32 | `e3f6+e5f3+cbatch_g4+pre_s45` | 0.814 / 0.835 | WIN / WIN |
| r1+r2 | 2:64:f32 | `e3c0+e5c1+cbatch+pre_s4` | 0.851 / 0.844 | WIN / WIN |
| r1+r2 | 2:128:f32 | `e3c4+cpre+pre_s4w2` | 0.913 / 0.917 | WIN / WIN |
| r3 | 1:64:f32 | `e3c1` | 0.931 / 0.933 | WIN / WIN |
| r3 | 1:128:f32 | `e3c1` | 0.953 / 0.961 | WIN / WIN |
| r3 | 2:8:f32 | `e3c1` | 0.976 / 0.981 | WIN / WIN |
| r3 | 2:64:f32 | `e3c1` | 0.948 / 0.943 | WIN / WIN |
| r4 | 1:5:f32 | `e5c0+cbatch_cpre+pre_s45` | 0.977 / 0.978 | WIN / WIN |
| r4 | 2:8:f32 | `lpad_entry` | 0.918 / 0.924 | WIN / WIN |
| r4 | 2:32:f32 | `e3f7+e5c0+lpad_entry` | 0.959 / 0.963 | WIN / WIN |
| r4 | 1:5:bf16 | `e5c0+cbatch_cpre+pre_s45` | 0.963 / 0.978 | WIN / WIN |
| r4 | 1:32:bf16 | `e3c1+cbatch_cpre+pre_s45` | 0.949 / 0.954 | WIN / WIN |
| r4 | 2:8:bf16 | `lpad_entry` | 0.900 / 0.904 | WIN / WIN |
| r4 | 2:16:bf16 | `e3c0` | 0.964 / 0.965 | WIN / WIN |
| r4 | 2:32:bf16 | `e3f7+e5c0+cbatch_cpre+lpad_entry` | 0.971 / 0.963 | WIN / WIN |
| r5 | 2:16:bf16 | `lpad_entry` | 0.913 / 0.928 | WIN / WIN |

## 4. Lever ledger by round (both nodes; W / T / L counted only when both nodes agree; every verdict is "on the round's base")

`round (base)`: r1+r2 = solo levers and combinations on the first bases (two jobs per node pooled); r3-r5 = the
accepted bases of the previous round, re-testing every lever (losers included) plus new combinations.

| round (base) | lever | arms x cells | median ratio | W / T / L (both nodes) |
|---|---|---|---|---|
| r1+r2 | E3 grid-fill plan | 152 | 1.080 (best 0.844) | 22 / 28 / 102 |
| r1+r2 | E3 planner plan | 87 | 1.022 (best 0.858) | 25 / 22 / 40 |
| r1+r2 | E5 grid-fill plan | 240 | 0.989 (best 0.884) | 82 / 98 / 60 |
| r1+r2 | E5 planner plan | 49 | 1.001 (best 0.879) | 10 / 25 / 14 |
| r1+r2 | cbatch | 12 | 0.984 (best 0.950) | 2 / 10 / 0 |
| r1+r2 | cbatch_cpre | 9 | 0.972 (best 0.954) | 3 / 6 / 0 |
| r1+r2 | cbatch_g4 | 12 | 0.981 (best 0.952) | 2 / 10 / 0 |
| r1+r2 | cepi | 8 | 0.947 (best 0.923) | 5 / 3 / 0 |
| r1+r2 | cepi_fit | 8 | 0.957 (best 0.921) | 5 / 3 / 0 |
| r1+r2 | combination | 732 | 0.864 (best 0.773) | 716 / 9 / 7 |
| r1+r2 | cpre | 9 | 0.989 (best 0.985) | 0 / 9 / 0 |
| r1+r2 | lpad | 11 | 0.998 (best 0.986) | 0 / 11 / 0 |
| r1+r2 | pre_s4 | 11 | 0.984 (best 0.959) | 1 / 10 / 0 |
| r1+r2 | pre_s45 | 12 | 0.985 (best 0.977) | 0 / 12 / 0 |
| r1+r2 | pre_s4w2 | 11 | 0.986 (best 0.958) | 3 / 7 / 1 |
| r1+r2 | pre_s5 | 11 | 1.003 (best 0.995) | 0 / 10 / 1 |
| r1+r2 | rec_stream | 6 | 1.062 (best 1.010) | 0 / 2 / 4 |
| r3 | E3 grid-fill plan | 162 | 1.145 (best 0.986) | 0 / 20 / 142 |
| r3 | E3 planner plan | 82 | 1.025 (best 0.932) | 9 / 32 / 41 |
| r3 | E5 grid-fill plan | 244 | 1.065 (best 0.986) | 0 / 61 / 183 |
| r3 | E5 planner plan | 55 | 1.069 (best 0.981) | 0 / 10 / 45 |
| r3 | acq_cta | 12 | 1.101 (best 1.027) | 0 / 0 / 12 |
| r3 | cbatch | 8 | 1.007 (best 0.999) | 0 / 8 / 0 |
| r3 | cbatch_cpre | 10 | 1.010 (best 0.992) | 0 / 9 / 1 |
| r3 | cbatch_g4 | 9 | 1.003 (best 0.999) | 0 / 9 / 0 |
| r3 | cepi | 1 | 0.963 (best 0.963) | 1 / 0 / 0 |
| r3 | cepi_fit | 1 | 0.962 (best 0.962) | 1 / 0 / 0 |
| r3 | cepi_lane | 12 | 1.009 (best 0.995) | 0 / 11 / 1 |
| r3 | combination | 350 | 1.020 (best 0.957) | 9 / 210 / 131 |
| r3 | cpre | 2 | 1.005 (best 1.003) | 0 / 2 / 0 |
| r3 | lpad_entry | 12 | 0.999 (best 0.991) | 0 / 12 / 0 |
| r3 | pre_none | 12 | 1.035 (best 1.018) | 0 / 5 / 7 |
| r3 | pre_s4 | 1 | 1.013 (best 1.013) | 0 / 1 / 0 |
| r3 | pre_s45 | 11 | 0.999 (best 0.996) | 0 / 11 / 0 |
| r3 | pre_s4w2 | 10 | 1.013 (best 0.998) | 0 / 9 / 1 |
| r3 | pre_s5 | 10 | 1.043 (best 1.015) | 0 / 4 / 6 |
| r3 | rec_stream | 7 | 1.071 (best 1.020) | 0 / 2 / 5 |
| r4 | E3 grid-fill plan | 326 | 1.194 (best 0.987) | 0 / 15 / 311 |
| r4 | E3 planner plan | 162 | 1.059 (best 0.955) | 2 / 55 / 105 |
| r4 | E5 grid-fill plan | 534 | 1.075 (best 0.987) | 0 / 107 / 427 |
| r4 | E5 planner plan | 118 | 1.072 (best 0.980) | 1 / 21 / 96 |
| r4 | acq_cta | 24 | 1.112 (best 1.026) | 0 / 1 / 23 |
| r4 | cbatch | 16 | 1.005 (best 0.993) | 0 / 14 / 2 |
| r4 | cbatch_cpre | 24 | 1.005 (best 0.984) | 0 / 22 / 2 |
| r4 | cbatch_g4 | 18 | 1.002 (best 0.992) | 0 / 17 / 1 |
| r4 | cepi | 2 | 0.998 (best 0.996) | 0 / 2 / 0 |
| r4 | cepi_fit | 2 | 0.996 (best 0.994) | 0 / 2 / 0 |
| r4 | cepi_lane | 18 | 1.005 (best 0.990) | 0 / 15 / 3 |
| r4 | combination | 652 | 1.007 (best 0.951) | 43 / 473 / 136 |
| r4 | cpre | 8 | 0.999 (best 0.988) | 0 / 8 / 0 |
| r4 | lpad_entry | 24 | 1.001 (best 0.902) | 2 / 22 / 0 |
| r4 | pre_none | 24 | 1.032 (best 1.005) | 0 / 10 / 14 |
| r4 | pre_s4 | 2 | 1.014 (best 1.013) | 0 / 2 / 0 |
| r4 | pre_s45 | 22 | 0.999 (best 0.984) | 0 / 22 / 0 |
| r4 | pre_s4w2 | 20 | 1.006 (best 0.985) | 0 / 17 / 3 |
| r4 | pre_s5 | 20 | 1.030 (best 1.005) | 0 / 10 / 10 |
| r4 | rec_stream | 14 | 1.063 (best 0.995) | 0 / 3 / 11 |
| r5 | E3 grid-fill plan | 125 | 1.223 (best 1.003) | 0 / 6 / 119 |
| r5 | E3 planner plan | 74 | 1.123 (best 0.973) | 0 / 14 / 60 |
| r5 | E5 grid-fill plan | 210 | 1.107 (best 1.012) | 0 / 17 / 193 |
| r5 | E5 planner plan | 58 | 1.134 (best 0.986) | 0 / 6 / 52 |
| r5 | acq_cta | 10 | 1.151 (best 1.074) | 0 / 0 / 10 |
| r5 | cbatch | 5 | 1.017 (best 1.000) | 0 / 3 / 2 |
| r5 | cbatch_cpre | 6 | 1.014 (best 0.993) | 0 / 4 / 2 |
| r5 | cbatch_g4 | 8 | 1.002 (best 0.994) | 0 / 6 / 2 |
| r5 | cepi_fit | 2 | 1.102 (best 1.075) | 0 / 0 / 2 |
| r5 | cepi_lane | 9 | 1.007 (best 0.998) | 0 / 7 / 2 |
| r5 | combination | 304 | 1.022 (best 0.975) | 0 / 159 / 145 |
| r5 | cpre | 1 | 1.004 (best 1.004) | 0 / 1 / 0 |
| r5 | lpad | 4 | 1.041 (best 1.002) | 0 / 2 / 2 |
| r5 | lpad_entry | 6 | 1.003 (best 0.921) | 1 / 5 / 0 |
| r5 | pre_none | 10 | 1.036 (best 1.014) | 0 / 4 / 6 |
| r5 | pre_s4 | 5 | 1.004 (best 1.002) | 0 / 5 / 0 |
| r5 | pre_s45 | 5 | 1.006 (best 0.997) | 0 / 4 / 1 |
| r5 | pre_s4w2 | 10 | 1.013 (best 1.002) | 0 / 10 / 0 |
| r5 | pre_s5 | 10 | 1.041 (best 1.013) | 0 / 2 / 8 |
| r5 | rec_stream | 8 | 1.068 (best 0.986) | 0 / 2 / 6 |
| r6 | E3 grid-fill plan | 35 | 1.164 (best 1.012) | 0 / 4 / 31 |
| r6 | E3 planner plan | 29 | 1.134 (best 0.970) | 0 / 6 / 23 |
| r6 | E5 grid-fill plan | 60 | 1.163 (best 1.026) | 0 / 1 / 59 |
| r6 | E5 planner plan | 28 | 1.167 (best 1.004) | 0 / 1 / 27 |
| r6 | acq_cta | 4 | 1.178 (best 1.142) | 0 / 0 / 4 |
| r6 | cbatch | 4 | 1.018 (best 1.007) | 0 / 3 / 1 |
| r6 | cbatch_cpre | 4 | 1.013 (best 1.006) | 0 / 3 / 1 |
| r6 | cbatch_g4 | 4 | 1.018 (best 1.001) | 0 / 3 / 1 |
| r6 | cepi_fit | 3 | 1.103 (best 1.074) | 0 / 0 / 3 |
| r6 | cepi_lane | 4 | 1.032 (best 1.006) | 0 / 1 / 3 |
| r6 | combination | 121 | 1.091 (best 0.980) | 0 / 34 / 87 |
| r6 | lpad | 3 | 1.103 (best 1.078) | 0 / 0 / 3 |
| r6 | lpad_entry | 1 | 1.003 (best 1.003) | 0 / 1 / 0 |
| r6 | pre_none | 4 | 1.031 (best 1.001) | 0 / 2 / 2 |
| r6 | pre_s45 | 4 | 1.012 (best 1.010) | 0 / 4 / 0 |
| r6 | pre_s4w2 | 4 | 1.018 (best 0.993) | 0 / 2 / 2 |
| r6 | pre_s5 | 4 | 1.045 (best 1.018) | 0 / 1 / 3 |
| r6 | rec_stream | 4 | 1.059 (best 0.997) | 0 / 2 / 2 |

parts of accepted changes: {'E3 planner plan': 13, 'E5 planner plan': 10, 'cepi_fit': 2, 'pre_s4': 9, 'E5 grid-fill plan': 5, 'cepi': 5, 'lpad': 1, 'cbatch_g4': 3, 'E3 grid-fill plan': 5, 'pre_s4w2': 2, 'pre_s45': 4, 'cbatch': 1, 'cpre': 1, 'cbatch_cpre': 4, 'lpad_entry': 5}
