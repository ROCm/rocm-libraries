# File Index

A by-file map of the `ck_dsl` package. Symbols listed are the primary public exports — see each file for the rest.

## Top Level

| Path | Primary contents |
|-------------------------------|--------------------------------------------------------------------------------------------------------|
| `__init__.py` | Top-level re-exports. The canonical "what does the DSL expose" file. |
| `__main__.py` | `python -m ck_dsl` lists discoverable entry points. |
| `transforms.py` | Coordinate-transform DAG: `CoordVar`, `Transform`, `PassThrough`, `Pad`, `PadDynamic`, `Embed`, `Merge`, `Unmerge`, `Indirect`, `TensorDescriptor`, constructors (`pass_through`, `pad`, `pad_dynamic`, `embed`, `merge`, `unmerge`, `indirect`). |
| `run_manifest.py` | `python -m ck_dsl.run_manifest`; numpy-backed HSACO + manifest runner. |
| `sweep.py` | Parallel build-on-the-fly sweep driver. `build_all_instances`, `build_default_dispatcher_set`, `write_sweep_manifest`. |
| `sweep_bench.py` | Benchmark driver consuming a sweep manifest; CSV output. |
| `torch_backend.py` | `torch.compile(fn, backend="ck_dsl")` entry point. |

## `core/` — IR and Lowering

| Path | Primary contents |
|-------------------------------|--------------------------------------------------------------------------------------------------------|
| `core/ir.py` | Typed SSA IR. `Type`, `VectorType`, `PtrType`, `SmemType`; singletons (`I1`, `I8`, `I32`, `I64`, `F16`, `BF16`, `F32`, `FP8E4M3`, `BF8E5M2`); cache-hint constants (`CACHE_ALL`, `CACHE_GLOBAL`, `CACHE_STREAM`, `NON_TEMPORAL`); `Value`, `Op`, `Region`, `Param`, `KernelDef`; `IRBuilder`. IRBuilder methods added in : `cvt_bf8_to_f32`, `cvt_f32_to_{fp8, bf8, i8_sat}`, `clamp_f32`, `global_atomic_add`, `lds_atomic_add`. |
| `core/ir_print.py` | `print_ir(kernel)` — MLIR-style textual IR dump. |
| `core/passes.py` | `optimize_kernel`, `canonicalize_region`, `eliminate_dead_pure_ops`, `PassStats`. Conservative constant fold + CSE + DCE. |
| `core/lower_llvm.py` | `lower_kernel_to_llvm(kernel) -> str`. Production AMDGPU LLVM IR emission. Datalayout, intrinsic declarations, op-to-IR mapping. |
| `core/lower_hip.py` | `lower_kernel_to_hip(kernel)` — HIP C++ debug emission. |
| `core/lower_cktile.py` | `lower_spec_to_cktile`, `lower_universal_gemm_to_cktile`, `lower_implicit_gemm_conv_to_cktile`. Spec-to-CK-Tile-C++ parity emitter. |

## `runtime/` — Build And Launch

| Path | Primary contents |
|-------------------------------|--------------------------------------------------------------------------------------------------------|
| `runtime/comgr.py` | `build_hsaco_from_llvm_ir(ir_text, isa=..., options=...) -> (bytes, ComgrTimings)`. Ctypes over `libamd_comgr`. |
| `runtime/hip_module.py` | `Runtime`, `Module`, `Event`, `HipError`. Ctypes over `libamdhip64`. Per-stream pending-args queue. |
| `runtime/torch_module.py` | `pack_args`, `pack_args_kernelparams`, `resolve_stream`, `empty_workspace`, `launch_torch_kernel`. |
| `runtime/launcher.py` | `KernelLauncher`, `PipelineLauncher`, `LaunchConfig`, `LaunchSummary`, `WorkspaceSpec`, `WorkspacePool`, `DeviceMem`, `time_launches`, `no_fence`, `synchronize_and_release`, `wait_stream_and_release`, `release_retained_for_stream`. |

## `helpers/` — Authoring Layer

| Path | Primary contents |
|---------------------------------------|-------------------------------------------------------------------------------------------------|
| `helpers/__init__.py` | Re-exports for the helper layer. |
| `helpers/atoms.py` | `MfmaAtom`, `MFMA_F16_ATOMS`, `MFMA_FP8_ATOMS`, `MFMA_ATOMS`, `mfma_atom(dtype, m, n, k)`. Nine atoms: f16 (4x4x4, 16x16x16, 16x16x32, 32x32x8, 32x32x16) + fp8 / bf8 (16x16x32, 32x32x16; ). |
| `helpers/geometry.py` | `WarpGrid` — block/warp/lane decomposition. |
| `helpers/loads.py` | `CoalescedTileLoader`, `AsyncTileLoader`, `AsyncTileLoaderSlot`, `DescriptorFn`, `lane_contiguous_descriptor`. |
| `helpers/layouts.py` | `LdsLayout`, `TransposeLdsReader`, `xor_swizzle_bytes`. |
| `helpers/schedule.py` | `SchedulePolicy`. `sched_group_barrier`, `s_setprio`, named schedule flavors. |
| `helpers/pipeline.py` | `SoftwarePipeline.run_ping_pong(...)`. |
| `helpers/epilogues.py` | `DirectEpilogue`, `CShuffleEpilogue`. |
| `helpers/attention.py` | `Attention2DConfig`, `Attention3DConfig`, `OnlineSoftmaxState`, `PagedKvDescriptor`, `apply_softcap_log2`, `apply_softcap_scalar`, `binary_search_seq_idx`, `causal_mask`, `sliding_window_mask`, `mfma_16x16x16_for_dtype`, `mfma_16x16x32_for_dtype`, `select_2d_config`, `select_3d_config`, `use_2d_kernel`, `warp_xor_reduce_max`, `warp_xor_reduce_sum`. |
| `helpers/compile.py` | `compile_kernel`, `KernelArtifact`. |
| `helpers/manifest.py` | `MANIFEST_SCHEMA = "ck.dsl.example.manifest/v1"`, `attention_args_signature`, `conv_args_signature`, `gemm_args_signature`, `make_attention_manifest`, `make_conv_manifest`, `make_gemm_manifest`, `make_simple_op_manifest`, `write_artifact`. |
| `helpers/tensor_view.py` | `TensorDescriptor`, `TensorView`, `TileWindow`, `TensorCoordinate`, `BufferResource`; constructors `make_global_view`, `make_lds_view`, `make_buffer_resource`, `make_buffer_view`, `make_naive_tensor_descriptor_packed`, `make_naive_tensor_view_packed`, `make_tile_window`, `make_tensor_coordinate`, `move_tensor_coordinate`, `view_from_transforms_descriptor`. |
| `helpers/distribution.py` | `TileDistributionEncoding`, `TileDistribution`, `StaticDistributedTensor`, `LoadStoreTraits`, `make_static_tile_distribution`, `make_load_store_traits`, `make_static_distributed_tensor`, `load_tile`, `store_tile`. |
| `helpers/sweep.py` | `sweep_row_chunks`, `pass2_row_chunks`, `RowChunkSweepResult`. |
| `helpers/io.py` | `io_ir_type`, `load_scalar`, `load_scalar_as_f32`, `load_vec`, `load_vec_as_f32`, `pack_f32_to`, `store_scalar`, `store_scalar_from_f32`, `store_vec`. |
| `helpers/reduction.py` | `ReduceCombine`, `block_lds_reduce`. |
| `helpers/spec.py` | `IOSpecRule`, `validate_io`, `SignatureBuilder`, `kernel_name_join`, `ceil_div_grid`, `ptr_type_str`, `sig_param`, `sig_scalar`. |
| `helpers/quant.py` | `QDType`, `QUANT_MAX_ABS`, `quant_ir_type`, `quant_max_abs`, `quantize_scalar_f32`, `dequantize_scalar_to_f32`, `ir_to_qdtype`. |
| `helpers/scan.py` | `lds_zero_i32`, `block_histogram_i32`, `block_exclusive_scan_i32`. Block-wide cooperative scan + histogram primitives. |
| `helpers/persistent.py` | `build_persistent_counter_init`, `persistent_tile_loop` (ctx-manager), `persistent_tile_for_each` (functional). Persistent-grid pattern shared with StreamK. |
| `helpers/streamk.py` | `StreamKPartition`, `StreamKReductionStrategy`, `compute_streamk_grid_size`, `emit_streamk_decode`, `streamk_num_macro_tiles`. Tile-id decode + grid sizing for the StreamK GEMM family. |
| `helpers/i4_dequant.py` | `unpack_i4_byte_to_pair_i32` / `_i8` / `_f32`, `dequant_i4_byte_to_fp8_pair`, `dequant_i4_byte_to_bf8_pair`. Packed-i4 weight unpack + sign-extend + scale-apply chain. |
| `helpers/mx_scale.py` | `decode_mx_scale_e8m0`, `apply_mx_scale`, `load_and_decode_mx_scale_byte`. OCP MX shared-exponent decode + per-block fmul. |
| `helpers/preshuffle.py` | `PreshuffleBSpec`, `emit_preshuffleb_offset`, `host_preshuffle_layout`. Tile-major B-layout descriptor for FP8 / BF8 / i4 GEMM bandwidth path. |
| `helpers/gather_scatter.py` | `gather_row_offset`, `scatter_token_offset`, `load_sorted_token_id`, `load_sorted_topk_weight`. Per-token indirect-address chain for the fused-MoE gather / topk-weighted scatter kernels; consumes the moe-sort outputs. |
| `helpers/rotary.py` | `RotaryLayout`, `RotarySpec`, `pair_indices`, `load_cos_sin`, `apply_rotary_pair_f32`. RoPE primitives for both interleaved (GPT-J) and LLaMA-half layouts; one (cos, sin) f32 pair per call, 2x2 rotation lowering to two `v_fma_f32`. |
| `helpers/rng.py` | `philox_u32_quartet`, `philox_uniform_f32_quartet`, `dropout_mask_pair_f32`, Philox constants (`PHILOX_M0/M1/W0/`, `PHILOX_ROUNDS`). Counter-based 10-round Philox4x32 RNG matching CK Tile + PyTorch bit-for-bit. |
| `helpers/attention.py` (extended) | `alibi_bias_log2`, `alibi_bias_matrix`, `custom_mask`. Three extra mask families to compose with the existing `causal_mask` / `sliding_window_mask`. |
| `helpers/qk_scale.py` | `QkScaleSpec`, `QkScaleLayout`, `load_q_scale_for_block`, `load_k_scale_for_block`, `apply_qk_scales`. Per-head or per-block Q/K scale loaders for Sage attention; one ``v_mul_f32`` per score after the AMDGPU combiner folds the scale-multiply pair. |
| `helpers/codebook.py` | `codebook_lookup_i8_to_fp8`, `codebook_lookup_i8_to_bf8`, `codebook_lookup_i4_pair_to_fp8`, `codebook_lookup_i4_pair_to_bf8`, `apply_per_tensor_scale`. i8 / i4 -> {fp8, bf8} codebook dequant chains for the Sage int variants; reuses 's `unpack_i4_byte_to_pair_i32` for the i4 unpack. |
| `helpers/sparse_iter.py` | `BlockSparseSpec`, `VsaSparseSpec`, `block_sparse_iter`, `vsa_lut_iter`, `load_block_count`. Block-sparse bitmap loop + variable-size attention LUT loop primitives shared between the Jenga and VSA sparse-attention kernels. |
| `helpers/grid.py` | `NUM_XCDS_MI300X`, `NUM_XCDS_MI325X`, `NUM_XCDS_MI350X`, `SuperTileSwizzleResult`, `chiplet_aware_super_tile`, `chiplet_aware_super_tile_dynamic`, `chiplet_transform_chunked`, `super_tile_swizzle`, etc. |
| `helpers/autotune.py` | `Autotuner`, `AutotuneConfig`, `AutotuneKey`, `AutotuneResult`, `autotune_sweep`, `make_autotune_key`, `spec_replace`. |
| `helpers/fuse.py` | Graph-level fusion driver: `compile_fn`, `explain_fn`, `fuse_matmul_bias_relu`, `EpilogueOp`, `BiasAdd`, `Cast`, `Clamp`, `GELU`, `ReLU`, `ResidualAdd`, `ResidualMul`, `Scale`, `SiLU`, `FusedEpilogue`, `FusionMatchError`, `FusionPlan`, `dtype_to_ir`, `ir_dtype_zero`, `ir_dtype_const`, `ir_dtype_global_load`. |
| `helpers/fusion_ir.py` | `FusionGraph`, `FusionOp`, `FusionRegion`, `FusionTensor`, `build_graph`. |
| `helpers/fusion_legalize.py` | `FusionLegalizer`, `LegalResult`. |
| `helpers/fusion_lowering.py` | `BuiltRegion`, `ElementwiseLowerer`, `ExplainOnlyLowerer`, `GemmEpilogueLowerer`, `LoweringRegistry`, `ReductionLowerer`, `default_lowering_registry`. |
| `helpers/fusion_memory.py` | `WorkspaceAllocation`, `WorkspacePlanner`, `materialize_plan`. |
| `helpers/fusion_scheduler.py` | `GreedyFusionScheduler`, `RegionCost`. |
| `helpers/fusion_validation.py` | `BackendTiming`, `BenchmarkCase`, `FusionMatrixRunner`, `ValidationReport`, `run_fusion_validation_matrix`. |

## `analysis/` And `benchmark/`

| Path | Primary contents |
|-----------------------------------|---------------------------------------------------------------------------------------------------|
| `analysis/ir.py` | `LlvmIrStats`, `analyze_llvm_ir(ir_text)`. Counts MFMA, async LDS, vector loads/stores, waitcnt, barriers. |
| `analysis/isa.py` | `HsacoAnalysis`, `IsaStats`, `ResourceInfo`, `analyze_hsaco`, `parse_isa`, `parse_resources`. |
| `analysis/report.py` | `VariantReport`, `compare_variant_reports`. |
| `benchmark/summary.py` | `BenchmarkSummary`, `summarize_runs`, `benchmark_manifest`. |

## `instances/` — Kernel Builders

| Path | Primary contents |
|-----------------------------------------|---------------------------------------------------------------------------------------------|
| `instances/gemm_universal.py` | `TileSpec`, `TraitSpec`, `DataSpec`, `UniversalGemmSpec`, `is_valid_spec`, `build_universal_gemm`, `all_dispatcher_configs`. |
| `instances/batched_gemm.py` | `BatchedGemmSpec`, `build_batched_gemm`, `batched_gemm_signature`, `batched_gemm_grid`. |
| `instances/grouped_gemm.py` | `GroupedGemmProblem`, `GroupedGemmSpec`, `build_grouped_gemm`, `grouped_gemm_signature`, `GroupedGemmLauncher`, `grouped_gemm_problems`. |
| `instances/conv_implicit_gemm.py` | `ConvProblem`, `ImplicitGemmConvSpec`, `make_a_descriptor`, `make_b_descriptor`, `make_d_descriptor`, `build_implicit_gemm_conv`. |
| `instances/conv_implicit_gemm_auto.py` | Experimental alternate implicit-GEMM builder. Not exported from `instances/__init__.py`. |
| `instances/conv_direct_grouped.py` | `DirectConvProblem`, `DirectConv16cSpec`, `DirectConv4cSpec`, `build_direct_conv_16c`, `build_direct_conv_4c`. |
| `instances/img2col.py` | `Img2ColSpec`, `build_img2col`, `img2col_grid`, `img2col_signature`. (CK Tile 04.) |
| `instances/pooling.py` | `PoolingProblem`, `Pooling2DSpec`, `PoolOp`, `build_pooling2d`, `pooling2d_grid`, `pooling2d_signature`. (CK Tile 36.) |
| `instances/permute_nd.py` | `PermuteSpec`, `build_permute`, `permute_grid`, `permute_signature`. Rank-up-to-8 n-D permute. (CK Tile 06.) |
| `instances/batched_transpose.py` | `BatchedTranspose2DSpec`, `build_batched_transpose2d`, `batched_transpose2d_grid`, `batched_transpose2d_signature`. (CK Tile 35.) |
| `instances/gemm_multi_d.py` | `GemmMultiDSpec`, `DOp`, `MAX_D`, `build_gemm_multi_d`, `gemm_multi_d_grid`, `gemm_multi_d_signature`. Variadic D operands via fused cshuffle epilogue. (CK Tile 19.) |
| `instances/gemm_multi_abd.py` | `GemmMultiAbdSpec`, `build_gemm_multi_abd`, `gemm_multi_abd_grid`, `gemm_multi_abd_signature`. Multi-A/B/D wrapper; multi-A/B load-combine = v2. (CK Tile 22.) |
| `instances/elementwise.py` | `ElementwiseSpec`, `build_elementwise`, `elementwise_grid`, `elementwise_signature`. (ext: silu / swish / tanh / sigmoid / quick_gelu, swiglu / geglu.) |
| `instances/reduce.py` | `Reduce2DSpec`, `ReduceOp`, `build_reduce2d`, `reduce2d_grid`, `reduce2d_signature`. (ext: min / prod.) |
| `instances/layernorm2d.py` | `LayerNorm2DSpec`, `build_layernorm2d`, `layernorm2d_grid`, `layernorm2d_signature`. |
| `instances/rmsnorm2d.py` | `RMSNorm2DSpec`, `build_rmsnorm2d`, `rmsnorm2d_grid`, `rmsnorm2d_signature`. |
| `instances/transpose.py` | `Transpose2DSpec`, `build_transpose2d`, `transpose2d_grid`, `transpose2d_signature`. |
| `instances/smoothquant.py` | `SmoothQuantSpec`, `build_smoothquant`, `smoothquant_grid`, `smoothquant_signature`. Per-row dynamic i8 / fp8e4m3 / bf8e5m2 quant. (CK Tile 12.) |
| `instances/moe_smoothquant.py` | `MoeSmoothQuantSpec`, `build_moe_smoothquant`, `moe_smoothquant_grid`, `moe_smoothquant_signature`. Per-expert variant via topk_ids. (CK Tile 14.) |
| `instances/add_rmsnorm2d_rdquant.py` | `AddRmsnorm2DRdquantSpec`, `build_add_rmsnorm2d_rdquant`, `add_rmsnorm2d_rdquant_grid`, `add_rmsnorm2d_rdquant_signature`. Fused (a+b) -> RMSNorm -> quant. (CK Tile 11.) |
| `instances/topk_softmax.py` | `TopkSoftmaxSpec`, `build_topk_softmax`, `topk_softmax_grid`, `topk_softmax_signature`. Tournament topk + softmax for MoE routers. (CK Tile 09.) |
| `instances/moe_sorting.py` | `MoeSortingSpec`, `build_moe_sort_{histogram, scan, scatter}`, `moe_sort_*_grid`, `moe_sort_*_signature`, `moe_sorting_workspace_bytes`. Three-launch pipeline (atomic histogram + LDS scan + atomic scatter). (CK Tile 13.) |
| `instances/flatmm.py` | `FlatMMSpec`, `build_flatmm`, `flatmm_grid`, `flatmm_signature`. v1 shares the batched_gemm body; the preshuffle-B variant lands with the `helpers/preshuffle.py`. (CK Tile 18.) |
| `instances/batched_contraction.py` | `BatchedContractionSpec`, `build_batched_contraction`, `batched_contraction_grid`, `batched_contraction_signature`, `flatten_batch_strides`. Rank-N leading-batch wrapper over batched_gemm. (CK Tile 41.) |
| `instances/streamk_gemm.py` | `StreamKGemmSpec`, `build_streamk_gemm`, `streamk_gemm_grid`, `streamk_gemm_signature`, `streamk_gemm_workspace_bytes`, `StreamKReductionStrategy` (re-export). v1 uses scalar inner GEMM + Atomic reduction strategy; MFMA upgrade and the Reduction strategy land as a follow-on. (CK Tile 40.) |
| `instances/block_scale_gemm.py` | `BlockScaleGemmSpec`, `QuantMode`, `MantissaDType`, `build_block_scale_gemm`, `block_scale_gemm_grid`, `block_scale_gemm_signature`. {a / b / abquant} x {fp8e4m3 / bf8e5m2} grouped-scale GEMM. v1 scalar inner; preshuffle-B + i4 + MFMA = v2 against the same spec. (CK Tile 38.) |
| `instances/mx_gemm.py` | `MxGemmSpec`, `MxMantissaDType`, `build_mx_gemm`, `mx_gemm_grid`, `mx_gemm_signature`. OCP MX E8M0 shared-exponent GEMM (group_k=32). v1 scalar inner with fp8 / bf8 mantissa; fp4 / fp6 mantissa = v2. (CK Tile 42.) |
| `instances/fused_moe.py` | `FusedMoeSpec`, `FusedMoeLauncher`, `build_moe_gather`, `build_moe_silu_mul`, `build_moe_topk_weighted_reduce`, `moe_*_grid`, `moe_*_signature`, `moe_fused_workspace_bytes`. The fused-MoE forward pipeline: gather + SwiGLU activation fusion + topk-weighted atomic reduce; per-expert GEMMs composed via `build_block_scale_gemm` / `build_universal_gemm` in the caller's runtime loop. (CK Tile 15.) |
| `instances/_fmha_common.py` | `FmhaCommonSpec`, `FmhaShape`, `FmhaMaskMode`, `validate_common_spec`, `fmha_fwd_inner_body`. Shared scaffolding for the FMHA family; the scalar-inner forward body factored so every variant (varlen / paged / GQA / fp8) shares one code path. |
| `instances/fmha_varlen.py` | `FmhaFwdVarlenSpec`, `build_fmha_fwd_varlen`, `fmha_fwd_varlen_grid`, `fmha_fwd_varlen_signature`. Variable-length forward driven by `cu_seqlens_q` / `cu_seqlens_k`; linear-scan find-seq-idx. v1 scalar; MFMA = v2. (CK Tile 01 varlen.) |
| `instances/fmha_appendkv.py` | `FmhaAppendKvSpec`, `build_fmha_fwd_appendkv`, `fmha_appendkv_grid`, `fmha_appendkv_signature`. KV-cache write kernel with optional rotary apply on K_new (V_new is never rotated). (CK Tile 01 appendkv.) |
| `instances/fmha_paged_prefill.py` | `FmhaFwdPagedPrefillSpec`, `build_fmha_fwd_paged_prefill`, `fmha_fwd_paged_prefill_grid`, `fmha_fwd_paged_prefill_signature`. Paged-KV during prefill with `block_table[seq_idx, :]` indirection; v1 scalar; MFMA = v2. (CK Tile 01 pagedkv_prefill.) |
| `instances/fmha_splitkv_decode.py` | `FmhaFwdSplitKvDecodeSpec`, `build_fmha_fwd_splitkv_decode_segment`, `build_fmha_fwd_splitkv_decode_reduce`, `fmha_fwd_splitkv_decode_segment_grid`, `fmha_fwd_splitkv_decode_reduce_grid`, matching `*_signature`. Two-launch split-KV decode (segment kernel + online-softmax reduce); the v2 hoist reuses the existing `attention_tiled_3d.py` body. (CK Tile 01 splitkv.) |
| `instances/fmha_head_grouping.py` | `FmhaFwdHeadGroupingSpec`, `build_fmha_fwd_head_grouping`, `fmha_fwd_head_grouping_grid`, `fmha_fwd_head_grouping_signature`. GQA / MQA forward (`HQ / HK >= 2`); the per-(batch, head, query) decomposition uses `block_id_z` for batch. (CK Tile 01 head_grouping.) |
| `instances/fmha_bwd.py` | `FmhaBwdSpec`, `build_fmha_bwd`, `fmha_bwd_grid`, `fmha_bwd_signature`. FMHA backward with `dQ` / `dK` / `dV` atomic fp32 accumulate (using `global_atomic_add`); recomputes softmax weights from saved `(M, L)`. (CK Tile 01 bwd.) |
| `instances/fmha_fwd_fp8.py` | `FmhaFwdFp8Spec`, `build_fmha_fwd_fp8`, `fmha_fwd_fp8_grid`, `fmha_fwd_fp8_signature`. FP8 (fp8e4m3 / bf8e5m2) KV cache forward with per-tensor `k_scale` / `v_scale`; uses `cvt_fp8_to_f32` / `cvt_bf8_to_f32` for the dequant. (CK Tile 01 fp8.) |
| `instances/sage_attention.py` | `SageAttentionSpec`, `SageQuantMode`, `build_sage_attention`, `sage_attention_grid`, `sage_attention_signature`. Single builder that dispatches on `quant_mode` to cover four CK Tile parity variants: `fp16_bf16` (baseline), `fp8_bf16`, `i8_fp8_bf16`, `i4_fp8_bf16`. Per-block Q+K scales from `helpers.qk_scale`; codebook dequant from `helpers.codebook` for the int variants. v1 scalar inner; MFMA v2 reuses FP8 atoms. (CK Tile 49.) |
| `instances/sparse_attention.py` | `JengaSparseSpec`, `VsaSparseSpec`, `build_jenga_sparse_attention`, `build_vsa_sparse_attention`, `*_grid`, `*_signature`. Block-sparse (Jenga; one-hot bitmap, skip zeroed K-blocks) and variable-size sparse (VSA; per-q LUT with `BlockCount[q_block]` bound). Both share a single scalar-inner per-K-block attention body factored within the file; the only difference is the K-iteration strategy from `helpers.sparse_iter`. (CK Tile 50.) |
| `instances/attention_unified.py` | `UnifiedAttentionProblem`, `UnifiedAttention2DSpec`, `UnifiedAttention3DSpec`, `UnifiedAttentionReduceSpec`, scalar 2D/3D builders, `run_unified_attention_torch`, `supports_native_unified_attention`, `attention_3d_workspace_nbytes`. |
| `instances/attention_tiled_2d.py` | `UnifiedAttention2DTiledSpec`, `build_unified_attention_2d_tiled`. |
| `instances/attention_tiled_3d.py` | `UnifiedAttention3DTiledSpec`, `UnifiedAttentionReduceTiledSpec`, `build_unified_attention_3d_tiled`, `build_unified_attention_reduce_tiled`. |

## `examples/`

| Path | Purpose |
|---------------------------------------------------|--------------------------------------------------------------------|
| `examples/bake_off_implicit_gemm.py` | Implicit-GEMM conv generator + manifest (used by `08_bake_off_implicit_gemm`). |
| `examples/bake_off_direct_conv_16c.py` | Direct grouped 16c generator (`09_bake_off_direct_conv_16c`). |
| `examples/bake_off_direct_conv_4c.py` | Direct grouped 4c generator (`10_bake_off_direct_conv_4c`). |
| `examples/distribution_reduce_demo.py` | 1D distribution-driven reduce demo. |
| `examples/distribution_2d_add_demo.py` | 2D distribution-driven add demo. |
| `examples/ck_tile_parity.py` | Small-op parity harness vs torch reference. |
| `examples/attention/parity_unified_attention.py` | Triton vs CK DSL attention parity harness (all paths). |

## `example/ck_tile/dsl/<N>_*/gen.py`

Each `gen.py` wraps a generator from `ck_dsl.examples` or `ck_dsl.instances`. The harness `python/test/test_ck_dsl_examples.py` discovers them automatically and runs `run_manifest --verify`.

| Example | Backing builder |
|-------------------------------------------------|------------------------------------------------|
| `02_layernorm2d` | `instances/layernorm2d.py` |
| `05_reduce` | `instances/reduce.py` |
| `06_gemm_universal_cshuffle` | `instances/gemm_universal.py` (hero) |
| `07_gemm_universal_sweep` | `instances/gemm_universal.py` (dispatcher set) |
| `08_bake_off_implicit_gemm` | `instances/conv_implicit_gemm.py` |
| `09_bake_off_direct_conv_16c` | `instances/conv_direct_grouped.py` (16c) |
| `10_bake_off_direct_conv_4c` | `instances/conv_direct_grouped.py` (4c) |
| `10_rmsnorm2d` | `instances/rmsnorm2d.py` |
| `16_batched_gemm` | `instances/batched_gemm.py` |
| `21_elementwise` | `instances/elementwise.py` |
| `37_transpose` | `instances/transpose.py` |

## Repo-Level Docs

| Path | Purpose |
|-----------------------------------------|----------------------------------------------------------------------|
| `python/ck_dsl/README.md` | Package README. |
| `python/ck_dsl/helpers/README.md` | Helper-layer reference (CK Tile parity table). |
| `python/ck_dsl/TRANSFORM_DAG.md` | Coordinate-transform DAG walkthrough (conv, paged attention). |
| `python/ck_dsl/RUNBOOK_COMPLIANCE.md` | Runbook section -> DSL primitive mapping, plus measured pass results. |
| `python/ck_dsl/dsl_docs/` | This documentation tree. |
| `python/ck_dsl/examples/attention/README.md` | Attention parity methodology and numbers. |
| `/workspace/OPTIMIZATION_RUNBOOK (1).md`| Long-form runbook (referenced throughout). |
