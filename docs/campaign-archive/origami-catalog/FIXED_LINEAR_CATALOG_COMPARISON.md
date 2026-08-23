# Unchanged fixed-linear selector: SK0 catalog vs SK3 catalog

Both libraries use the exact same frozen 22-feature model and runtime code. The only difference is the included catalog:

- **Linear-SK0:** 298 production SK0 kernels.
- **Linear-SK3:** 192 Resource/Edge-campaign SK3 kernels.

Neither GridBased nor Origami selects a kernel in this experiment. `FixedLinearCatalogLibrary` scores every predicate-valid candidate and launches the maximum-score kernel.

## 1,500-shape, 3-repetition paired result

Ratio is Linear-SK3 / Linear-SK0.

| Metric | Result |
|---|---:|
| Geometric mean | **100.81%** |
| Arithmetic mean | **103.70%** |
| Median | **98.96%** |
| P10 / P5 | **84.39% / 77.42%** |
| Minimum | **40.72%** |
| P90 / P95 / maximum | **114.18% / 148.18% / 517.35%** |
| SK3 wins | 561 / 1500 |
| SK3 >=10% faster | 194 |
| SK3 below 90% | 241 |
| SK3 below 80% | 95 |

## Key strata

| Stratum | Geomean | Median | P10 | Minimum |
|---|---:|---:|---:|---:|
| Both M,N <1000 | **117.37%** | 100.90% | 90.44% | 70.14% |
| Both M,N <64 | **131.53%** | 103.70% | 99.08% | 97.22% |
| GEMV-like | **112.94%** | 102.86% | 92.78% | 55.70% |
| Skinny-N | **95.91%** | 97.90% | 76.72% | 40.72% |
| Skinny-M | **100.23%** | 98.68% | 85.43% | 64.15% |
| Interpolation | **100.80%** | 99.00% | 85.28% | 55.70% |
| Extrapolation | **100.85%** | 98.81% | 82.04% | 40.72% |

## Interpretation

With selection held identical, the SK3 catalog is 0.81% faster geometrically and 3.70% faster arithmetically, driven by very strong tiny/deep and GEMV wins. Typical performance is slightly lower (median -1.04%) and the tail remains significantly worse. Skinny-N remains the clearest weak class.

This comparison isolates catalog/execution-mode differences under the unchanged linear model more cleanly than the earlier GridBased-vs-Origami system comparison.

Artifacts:

- `measurements/linear_catalog_compare.csv`
- `reports/linear_sk0_vs_linear_sk3.json`
- `device_libs/linear_sk0/library/gfx1100`
- `device_libs/linear_sk3/library/gfx1100`
