# Custom MXFP8 / FP8 Compute + Memory GEMM (gfx1250)

Prebuilt Tensile Lite clients for these custom GEMM kernels on **gfx1250**
(compute-bound MX, memory-bound MX, and non-MX FP8). Each custom kernel tree is
self-contained: run the checked-in `tensile_client` against the bundled library
code objects — no hipBLASLt library rebuild is required.

**Inventory:** 12 custom kernel trees.

## How to run (any custom kernel)

From the tensilelite directory that contains the `custom_*` folders:

```bash
KD=<custom_kernel_directory>
# Discover the ClientParameters.ini path:
INI=$(find "$KD" -name ClientParameters.ini | head -1)
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/llvm/lib/ \
  ./"$KD"/0_Build/client/tensile_client --config-file "$INI"
```

Each tree contains:

- `0_Build/client/tensile_client` — prebuilt client binary
- `1_BenchmarkProblems/.../00_Final/source/ClientParameters.ini` — run config
- `1_BenchmarkProblems/.../00_Final/source/library/` — `TensileLibrary.yaml`, `.co` (and `.hsaco` when present)
- Assembly source under `.../source/build_tmp/SOURCE/assembly/` (reference)

## Prerequisites

- ROCm with gfx1250 support (`/opt/rocm`)
- `libomp` available via the LLVM runtime path used below
- Working directory: the **tensilelite** directory that contains the `custom_*` folders
  (paths in `ClientParameters.ini` are relative to that cwd)

## Configure

Shipped `ClientParameters.ini` values follow the source defaults (compute
kernels typically use high enqueue counts; memory kernels typically use lower
counts and random inits). For a quick smoke / correctness check, temporarily set
`num-elements-to-validate` to a small positive value (for example `128`) and lower
`num-enqueues-per-sync` (for example to `1`–`10`).

## Compute-bound (6)

### MXFP8 × MXFP8

| Custom kernel | Size (M,N,B,K) | Problem type |
|---------------|----------------|--------------|
| `custom_MXFP8xMXFP8_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep` | `8192,8192,1,8192` | `Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH` |
| `custom_MXFP8xMXFP8_BS1_8Kx8Kx4K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep` | `8192,8192,1,4096` | `Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH` |
| `custom_MXFP8xMXFP8_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore` | `8192,8192,1,8192` | `Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH` |
| `custom_MXFP8xMXFP8_BS1_8Kx8Kx4K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore` | `8192,8192,1,4096` | `Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH` |

### MXFP8 × MXFP4

| Custom kernel | Size (M,N,B,K) | Problem type |
|---------------|----------------|--------------|
| `custom_MXFP8xMXFP4_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore` | `8192,8192,1,8192` | `Cijk_Alik_Bljk_F8F4F8S_MXAE8B32_MXBE8B32_BH` |
| `custom_MXFP8xMXFP4_BS1_8Kx8Kx4K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore` | `8192,8192,1,4096` | `Cijk_Alik_Bljk_F8F4F8S_MXAE8B32_MXBE8B32_BH` |

Example:

```bash
KD=custom_MXFP8xMXFP8_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore
INI=$(find "$KD" -name ClientParameters.ini | head -1)
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/llvm/lib/ \
  ./"$KD"/0_Build/client/tensile_client --config-file "$INI"
```

## Memory-bound MXFP8 × MXFP8 (2)


| Custom kernel | Size (M,N,B,K) | Problem type |
|---------------|----------------|--------------|
| `custom_MXFP8_BS128_64x5kx2k_Persist_to_loop_spreadTDM_PKAReorder_PERMUTE_NT_fuseMXA_HiLnMX_TDMSched` | `64,5120,128,2048` | `Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH` |
| `custom_MXFP8_BS128_64x5kx1k_Persist_to_loop_spreadTDM_PKAReorder_PERMUTE_NT_fuseMXA_HiLnMX_TDMSched_MXSATileVgpr_halfATileVgpr` | `64,5120,128,1024` | `Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH` |

## Memory-bound MXFP8 × MXFP4 (2)

| Custom kernel | Size (M,N,B,K) | Problem type |
|---------------|----------------|--------------|
| `custom_MXFP8xMXFP4_BS128_64x5Kx2K_clean_spreadTDM_Persist_to_loop_barrier_before_buffer_store_VWA2_PERMUTE_NT_Reorder_interleave_PKAReorder_TDMBarrier_fuseMXA_BC0_opt_2DU_HiLnMX` | `64,5120,128,2048` | `Cijk_Alik_Bljk_F8F4F8S_MXAE8B32_MXBE8B32_BH` |
| `custom_MXFP8xMXFP4_BS128_64x5Kx1K_clean_spreadTDM_Persist_to_loop_barrier_before_buffer_store_VWA2_PERMUTE_NT_Reorder_interleave_PKAReorder_ATileVgpr_TDMBarrier_fuseMXA_BC0_MXSATileVgpr_opt_rebalance_2DU_HiLnMX` | `64,5120,128,1024` | `Cijk_Alik_Bljk_F8F4F8S_MXAE8B32_MXBE8B32_BH` |

## Non-MX FP8 × FP8 (2)


| Custom kernel | Size (M,N,B,K) | Problem type |
|---------------|----------------|--------------|
| `custom_FP8xFP8_BS1_4Kx4Kx64K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep` | `4096,4096,1,65536` | `Cijk_Alik_Bljk_F8F8S_BH` |
| `custom_FP8xFP8_BS1_4Kx4Kx64K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore` | `4096,4096,1,65536` | `Cijk_Alik_Bljk_F8F8S_BH` |
