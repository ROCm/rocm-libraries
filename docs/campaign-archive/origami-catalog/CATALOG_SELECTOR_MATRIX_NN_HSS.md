# Catalog x selector matrix — gfx1100, stock Origami

All values are **geometric means of per-shape ratios to that target's production baseline**, so each `g0` row is 1.000 by definition.

`Prediction` sets `[7]` to null and discards production's exact table; `GridBased` is a Matching library that selects by nearest neighbour over it.


## HSS-TN  (1511 evaluation shapes)

### Overall

| catalog | table type | selector | kernels | geomean | P10 | off-table | on-table |
|---|---|---|---:|---:|---:|---:|---:|
| production GridBased | GridBased | nearest-neighbour matching | 3 | **1.000** | 1.000 | 1.000 | 1.000 |
| identity collapse | Prediction | stock Origami | 105 | **1.116** | 0.781 | 1.117 | 0.936 |
| subset search | Prediction | stock Origami | 85 | **1.153** | 0.812 | 1.154 | 0.925 |
| subset search, tier-balanced | Prediction | stock Origami | 72 | **1.177** | 0.862 | 1.178 | 0.962 |

### By geometry (aspect ratio of M x N)

| catalog | table type | kernels | gemv | skinny | rect | square |
|---|---|---:|---:|---:|---:|---:|
| production GridBased | GridBased | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| identity collapse | Prediction | 105 | 1.392 | 1.179 | 1.071 | 1.069 |
| subset search | Prediction | 85 | 1.416 | 1.214 | 1.111 | 1.104 |
| subset search, tier-balanced | Prediction | 72 | 1.425 | 1.243 | 1.133 | 1.128 |

### By size (output tier, M x N)

| catalog | table type | kernels | tiny | small | medium | large |
|---|---|---:|---:|---:|---:|---:|
| production GridBased | GridBased | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| identity collapse | Prediction | 105 | 1.377 | 1.158 | 0.961 | 1.036 |
| subset search | Prediction | 85 | 1.436 | 1.195 | 0.987 | 1.067 |
| subset search, tier-balanced | Prediction | 72 | 1.430 | 1.202 | 1.030 | 1.109 |

## HHS-NN  (1914 evaluation shapes)

### Overall

| catalog | table type | selector | kernels | geomean | P10 | off-table | on-table |
|---|---|---|---:|---:|---:|---:|---:|
| production GridBased | GridBased | nearest-neighbour matching | 70 | **1.000** | 1.000 | 1.000 | 1.000 |
| identity collapse | Prediction | stock Origami | 100 | **0.993** | 0.718 | 0.990 | 1.001 |
| subset search, tier-balanced | Prediction | stock Origami | 72 | **1.029** | 0.853 | 1.031 | 1.022 |

### By geometry (aspect ratio of M x N)

| catalog | table type | kernels | gemv | skinny | rect | square |
|---|---|---:|---:|---:|---:|---:|
| production GridBased | GridBased | 70 | 1.000 | 1.000 | 1.000 | 1.000 |
| identity collapse | Prediction | 100 | 1.314 | 1.016 | 0.979 | 0.974 |
| subset search, tier-balanced | Prediction | 72 | 1.304 | 1.076 | 1.009 | 1.003 |

### By size (output tier, M x N)

| catalog | table type | kernels | tiny | small | medium | large |
|---|---|---:|---:|---:|---:|---:|
| production GridBased | GridBased | 70 | 1.000 | 1.000 | 1.000 | 1.000 |
| identity collapse | Prediction | 100 | 1.090 | 1.008 | 0.940 | 0.960 |
| subset search, tier-balanced | Prediction | 72 | 1.123 | 1.038 | 0.978 | 1.005 |

## Definitions

- **gemv** min(M,N)=1 · **skinny** max>=8x min · **rect** max>=2x min · **square** otherwise
- **tiny** M*N < 256² · **small** < 1024² · **medium** < 4096² · **large** >= 4096²
- **P10** is the 10th percentile of the per-shape ratio (worst tenth), not the mean of a decile.
