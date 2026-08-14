# bf16 Fused Epilogue Performance Study (gfx950)

Isolates the per-component cost of the fused RMSNorm epilogue for the bf16-in /
bf16-out K1 winner kernel by toggling the three epilogue flags and benchmarking
each variant on a gfx950 GPU.

## Setup

- **GPU:** gfx950 (CDNA4)
- **Problem:** GEMM `8192 x 8192 x 8192`, batch 1, bf16 in / bf16 out, f32 compute (`TransposeA=True`, `TransposeB=False`, `UseBeta=False`).
- **Winner tile (identical across all variants):** MT 384x256x64, MI 16x16x32, MIWaveTile [12,8], MIWaveGroup [2,2], WorkGroup [32,8,1], DepthU=64, PrefetchGlobalRead=2, StreamK=3, `UseSubtileImpl=True`. Only the epilogue flags differ between variants, so the accumulation loop is identical.
- **Metric:** GPU kernel time in microseconds (`KernelTime: True`), reported by the Tensile client. `TotalFlops = 1.0995e12`; `us = TotalFlops / (GFlops x 1000)`.
- **Run conditions:** all kernel/asm caches and output directories were removed before the run, so every kernel was rebuilt from scratch. The full-epilogue variant additionally validated against a CPU reference (`NumElementsToValidate: -1`) and PASSED.

### Epilogue flags

| Flag | Effect |
|---|---|
| `PartialRMS` | Master switch: per-row sum-of-squares reduction + gamma. |
| `PartialRMSResidualAdd` | Load the residual tensor, compute `H = GEMM + residual`, write `H` back to the accumulators. |
| `PartialRMSStoreBf16D` | Store `H` as bf16 to the `ResidualOut` tensor (the "store-back"). |

## Configurations

| Config | YAML | PartialRMS | ResidualAdd | StoreBack |
|---|---|---|---|---|
| Baseline (no epilogue) | `YAMLs/baseline_bf16_k1_winner.yaml` | – | – | – |
| PartialRMS only | `YAMLs/noresidual_bf16_k1_winner.yaml` | T | F | F |
| PartialRMS + ResidualAdd (store-back off) | `YAMLs/residual_nostoreback_bf16_k1_winner.yaml` | T | T | F |
| Full epilogue | `YAMLs/epilogues_bf16_k1_winner.yaml` | T | T | T |

## Results

| Configuration | Time (us) | TFLOPS | vs Baseline |
|---|---|---|---|
| Baseline (no epilogue) | 770.5 | 1427 | — |
| PartialRMS only (residual off) | 783.3 | 1404 | +12.8 us (+1.7%) |
| PartialRMS + ResidualAdd (store-back off) | 835.2 | 1316 | +64.7 us (+8.4%) |
| Full epilogue (PartialRMS + ResidualAdd + StoreBack) | 882.3 | 1246 | +111.8 us (+14.5%) |

### Marginal cost per component (incremental, added on top of the row above)

| Component | Delta time | % of baseline |
|---|---|---|
| PartialRMS sum-of-squares reduction | +12.8 us | +1.7% |
| ResidualAdd (residual load + add + write-back) | +51.9 us | +6.7% |
| ResidualOut bf16 store-back | +47.1 us | +6.1% |
| **Total fused-epilogue overhead** | **+111.8 us** | **+14.5%** |

## Interpretation

- **PartialRMS is nearly free (~1.7%, ~13us).** The sum-of-squares reduction operates on accumulator registers that are already resident, plus a tiny per-row `partialBuf` write. It adds no full-tensor memory traffic, so it barely moves kernel time.
- **ResidualAdd is the single most expensive component (~6.7%, ~52us).** It adds a full `M x N` residual tensor load from global memory (`8192 x 8192 x bf16` = 134 MB), then adds and writes back into the accumulators. Implied effective bandwidth ~ 134 MB / 52 us ~ 2.6 TB/s — consistent with a memory-bound extra HBM pass.
- **The store-back is almost as expensive as ResidualAdd (~6.1%, ~47us).** It writes the full `M x N` bf16 `ResidualOut` tensor (another 134 MB). ~ 134 MB / 47 us ~ 2.85 TB/s — again memory-bound. The store-back is *not* negligible.
- **Dominant cost is memory traffic, not compute.** ResidualAdd + StoreBack together account for ~12.8% of the ~14.5% total, and both are extra full-tensor HBM passes. The actual RMSNorm math (PartialRMS) contributes only ~1.7%.

## Note on the infeasible fifth configuration

"ResidualAdd on, PartialRMS off" cannot be built via YAML. Residual add is a sub-feature of the PartialRMS epilogue:

- `Tensile/SolutionStructs/Solution.py:327` rejects `PartialRMSResidualAdd` without `PartialRMS`.
- `Tensile/KernelWriterAssembly.py:15297-15304` gates both the residual emitter and the sum-of-squares pass inside `if kernel["PartialRMS"]:`.
- `Tensile/KernelWriter.py:10346` gates the `ResidualBuf` / `AddressResidualOut` kernargs the same way.

Isolating "residual without RMS reduction" would require a code change to add a residual-only path. The `PartialRMS only` config already isolates the reduction in the opposite direction (it costs ~1.7%).

## Reproduction

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite
./Tensile/bin/Tensile epilogues/YAMLs/baseline_bf16_k1_winner.yaml            /tmp/perf_baseline_bf16
./Tensile/bin/Tensile epilogues/YAMLs/noresidual_bf16_k1_winner.yaml          /tmp/perf_noresidual_bf16
./Tensile/bin/Tensile epilogues/YAMLs/residual_nostoreback_bf16_k1_winner.yaml /tmp/perf_residual_nostoreback_bf16
./Tensile/bin/Tensile epilogues/YAMLs/epilogues_bf16_k1_winner.yaml           /tmp/perf_full_bf16
```

The measured GFlops is the per-solution column in `2_BenchmarkData/*_00.csv`; convert to microseconds with `us = 1099511627776 / (GFlops x 1000)`.
