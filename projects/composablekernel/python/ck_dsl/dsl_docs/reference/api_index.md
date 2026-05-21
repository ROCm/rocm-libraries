# API Index

Top-level re-exports from `ck_dsl` and `ck_dsl.helpers`. Use this as a quick lookup when reading other docs.

## `from ck_dsl import ...`

```text
# Core types
BF16, BF8E5M2, F16, F32, FP8E4M3, I1, I8, I32, I64
IRBuilder, KernelDef, Op, Param, Region, Type, Value
PtrType, SmemType, VectorType

# Printing / lowering
print_ir
optimize_kernel, PassStats
lower_kernel_to_llvm
lower_kernel_to_hip
lower_spec_to_cktile
lower_universal_gemm_to_cktile
lower_implicit_gemm_conv_to_cktile

# Runtime / build
ComgrError, ComgrTimings, build_hsaco_from_llvm_ir
HipError, Runtime

# Authoring helpers (re-exported)
compile_kernel, KernelArtifact
MfmaAtom, mfma_atom
MFMA_ATOMS, MFMA_F16_ATOMS, MFMA_FP8_ATOMS # : fp8 + bf8 atoms
WarpGrid
CoalescedTileLoader, AsyncTileLoader, AsyncTileLoaderSlot
DirectEpilogue, CShuffleEpilogue
LdsLayout
SchedulePolicy
SoftwarePipeline

# Manifests
make_gemm_manifest, make_conv_manifest, make_attention_manifest
attention_args_signature, conv_args_signature, gemm_args_signature
write_artifact

# Attention helpers
Attention2DConfig, Attention3DConfig
OnlineSoftmaxState, PagedKvDescriptor
apply_softcap_scalar, causal_mask, sliding_window_mask
select_2d_config, select_3d_config, use_2d_kernel

# Transforms (CK Tile coord-DAG)
CoordVar, Indirect, PadDynamic, TensorDescriptor
pass_through, pad, pad_dynamic, embed, merge, unmerge, indirect

# Analysis / benchmark
BenchmarkSummary, benchmark_manifest, summarize_runs
HsacoAnalysis, IsaStats, ResourceInfo, VariantReport, LlvmIrStats
analyze_hsaco, analyze_llvm_ir, compare_variant_reports
parse_isa, parse_resources
```

## `from ck_dsl.helpers import ...` (extras not in top-level)

```text
# Tensor views
TensorDescriptor, TensorView, TileWindow, TensorCoordinate, BufferResource
make_global_view, make_lds_view
make_buffer_resource, make_buffer_view
make_naive_tensor_descriptor_packed, make_naive_tensor_view_packed
make_tile_window
make_tensor_coordinate, move_tensor_coordinate
view_from_transforms_descriptor

# Tile distributions
TileDistributionEncoding, TileDistribution
StaticDistributedTensor, LoadStoreTraits
make_static_tile_distribution, make_load_store_traits
make_static_distributed_tensor
load_tile, store_tile

# Sweeps
sweep_row_chunks, pass2_row_chunks, RowChunkSweepResult

# I/O
io_ir_type
load_scalar, load_scalar_as_f32
load_vec, load_vec_as_f32
pack_f32_to
store_scalar, store_scalar_from_f32, store_vec

# Reductions
ReduceCombine, block_lds_reduce # ext: "min" and "prod"
 # added to the combiner set

# MoE infra: histogram + exclusive scan + LDS zero-init
lds_zero_i32
block_histogram_i32
block_exclusive_scan_i32

# MoE / StreamK shared: persistent-kernel pattern
build_persistent_counter_init
persistent_tile_loop
persistent_tile_for_each

# StreamK partitioner (helpers/streamk.py)
StreamKPartition, StreamKReductionStrategy
compute_streamk_grid_size, emit_streamk_decode, streamk_num_macro_tiles

# helpers
# i4 packed-weight dequant (helpers/i4_dequant.py)
unpack_i4_byte_to_pair_f32, unpack_i4_byte_to_pair_i32, unpack_i4_byte_to_pair_i8
dequant_i4_byte_to_fp8_pair, dequant_i4_byte_to_bf8_pair
# OCP MX E8M0 shared-exponent (helpers/mx_scale.py)
decode_mx_scale_e8m0, apply_mx_scale, load_and_decode_mx_scale_byte
# Preshuffled-B layout (helpers/preshuffle.py)
PreshuffleBSpec, emit_preshuffleb_offset, host_preshuffle_layout
# FP8 / BF8 MFMA catalog (helpers/atoms.py)
MFMA_FP8_ATOMS, MFMA_ATOMS

# helpers
# Per-token indirect address calculation (helpers/gather_scatter.py)
gather_row_offset # sorted_token_ids[bucket]*hidden + col
scatter_token_offset # token_id*hidden + col (write-side counterpart)
load_sorted_token_id # i32 global load with align=4
load_sorted_topk_weight # f32 global load with align=4

# helpers
# Rotary position embedding (helpers/rotary.py)
RotaryLayout, RotarySpec # "interleaved" / "half"
pair_indices, load_cos_sin, apply_rotary_pair_f32
# Philox4x32-10 RNG for dropout (helpers/rng.py)
PHILOX_M0, PHILOX_M1, PHILOX_W0, PHILOX_W1, PHILOX_ROUNDS
philox_u32_quartet # 4 u32 from one counter
philox_uniform_f32_quartet # 4 f32 in [0, 1)
dropout_mask_pair_f32 # (keep, scale, keep, scale)
# Extended attention masks (helpers/attention.py)
alibi_bias_log2 # slope * (k_pos - q_pos)
alibi_bias_matrix # precomputed (H, Q, K) bias
custom_mask # per-(Q, K) i8 bool mask

# helpers
# Per-head / per-block Q + K scales for Sage attention (helpers/qk_scale.py)
QkScaleLayout, QkScaleSpec # "per_head" / "per_block"
load_q_scale_for_block, load_k_scale_for_block, apply_qk_scales
# Codebook dequant for sage int variants (helpers/codebook.py)
codebook_lookup_i8_to_fp8, codebook_lookup_i8_to_bf8
codebook_lookup_i4_pair_to_fp8, codebook_lookup_i4_pair_to_bf8
apply_per_tensor_scale
# Sparse attention K-iterators (helpers/sparse_iter.py)
BlockSparseSpec, VsaSparseSpec
block_sparse_iter # bitmap-driven
vsa_lut_iter # indirect-LUT
load_block_count # BlockCount[q_block]

# Spec helpers
IOSpecRule, validate_io
SignatureBuilder, kernel_name_join
ceil_div_grid, ptr_type_str, sig_param, sig_scalar

# Quantization (helpers/quant.py)
QDType, QUANT_MAX_ABS
quantize_scalar_f32, dequantize_scalar_to_f32
quant_ir_type, quant_max_abs, ir_to_qdtype

# Chiplet swizzle (helpers/grid.py)
NUM_XCDS_MI300X, NUM_XCDS_MI325X, NUM_XCDS_MI350X
SuperTileSwizzleResult
chiplet_aware_super_tile, chiplet_aware_super_tile_dynamic
chiplet_transform_chunked, chiplet_transform_chunked_dynamic
super_tile_swizzle, super_tile_swizzle_dynamic

# Autotune (helpers/autotune.py)
Autotuner, AutotuneConfig, AutotuneKey, AutotuneResult
autotune_sweep, make_autotune_key, spec_replace

# Fusion (helpers/fuse.py)
compile_fn, explain_fn, fuse_matmul_bias_relu
EpilogueOp, BiasAdd, Cast, Clamp, GELU, ReLU, ResidualAdd, ResidualMul, Scale, SiLU
FusedEpilogue, FusionMatchError, FusionPlan
dtype_to_ir, ir_dtype_zero, ir_dtype_const, ir_dtype_global_load

# Fusion infrastructure
FusionGraph, FusionOp, FusionRegion, FusionTensor, build_graph
FusionLegalizer, LegalResult
BuiltRegion, ElementwiseLowerer, ExplainOnlyLowerer, GemmEpilogueLowerer
LoweringRegistry, ReductionLowerer, default_lowering_registry
WorkspaceAllocation, WorkspacePlanner, materialize_plan
GreedyFusionScheduler, RegionCost
BackendTiming, BenchmarkCase, FusionMatrixRunner, ValidationReport, run_fusion_validation_matrix

# Manifest schema
MANIFEST_SCHEMA # "ck.dsl.example.manifest/v1"
make_simple_op_manifest
```

## `from ck_dsl.runtime.launcher import ...`

```text
KernelLauncher, PipelineLauncher
LaunchConfig, LaunchSummary
WorkspaceSpec, WorkspacePool
DeviceMem
time_launches, no_fence
synchronize_and_release
wait_stream_and_release
release_retained_for_stream
```

## `from ck_dsl.runtime.torch_module import ...`

```text
pack_args, pack_args_kernelparams
resolve_stream
empty_workspace
launch_torch_kernel # back-compat shim
```

## `from ck_dsl.instances import ...`

```text
# GEMM family
TileSpec, TraitSpec, DataSpec, UniversalGemmSpec, build_universal_gemm
BatchedGemmSpec, build_batched_gemm
GroupedGemmProblem, GroupedGemmSpec, build_grouped_gemm
GroupedGemmLauncher, grouped_gemm_problems

# Convolution family
ConvProblem, ImplicitGemmConvSpec, build_implicit_gemm_conv
DirectConvProblem, DirectConv16cSpec, build_direct_conv_16c
DirectConv4cSpec, build_direct_conv_4c
Img2ColSpec, build_img2col
PoolingProblem, Pooling2DSpec, PoolOp, build_pooling2d

# Attention family
UnifiedAttentionProblem
UnifiedAttention2DSpec, build_unified_attention_2d
UnifiedAttention3DSpec, build_unified_attention_3d
UnifiedAttentionReduceSpec, build_unified_attention_reduce
UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled
UnifiedAttention3DTiledSpec, build_unified_attention_3d_tiled
UnifiedAttentionReduceTiledSpec, build_unified_attention_reduce_tiled
run_unified_attention_torch
supports_native_unified_attention
attention_3d_workspace_nbytes

# Small ops
ElementwiseSpec, build_elementwise # ext: swiglu / geglu /
 # tanh / sigmoid / quick_gelu
Reduce2DSpec, ReduceOp, build_reduce2d # ext: min / prod
LayerNorm2DSpec, build_layernorm2d
RMSNorm2DSpec, build_rmsnorm2d
Transpose2DSpec, build_transpose2d

# small ops
BatchedTranspose2DSpec, build_batched_transpose2d # CK Tile 35
GemmMultiDSpec, build_gemm_multi_d # CK Tile 19
GemmMultiAbdSpec, build_gemm_multi_abd # CK Tile 22
PermuteSpec, build_permute # CK Tile 06

# quantisation kernels (i8 / fp8e4m3 / bf8e5m2 outputs)
SmoothQuantSpec, build_smoothquant # CK Tile 12
MoeSmoothQuantSpec, build_moe_smoothquant # CK Tile 14
AddRmsnorm2DRdquantSpec, build_add_rmsnorm2d_rdquant # CK Tile 11
TopkSoftmaxSpec, build_topk_softmax # CK Tile 09

# MoE sorting (three-kernel pipeline)
MoeSortingSpec
build_moe_sort_histogram, moe_sort_histogram_grid, moe_sort_histogram_signature
build_moe_sort_scan, moe_sort_scan_grid, moe_sort_scan_signature
build_moe_sort_scatter, moe_sort_scatter_grid, moe_sort_scatter_signature
moe_sorting_workspace_bytes

# GEMM variants
FlatMMSpec, build_flatmm, flatmm_grid, flatmm_signature # CK Tile 18
BatchedContractionSpec, build_batched_contraction # CK Tile 41
batched_contraction_grid, batched_contraction_signature, flatten_batch_strides
StreamKGemmSpec, build_streamk_gemm # CK Tile 40
streamk_gemm_grid, streamk_gemm_signature, streamk_gemm_workspace_bytes
StreamKReductionStrategy # enum

# quantised GEMM family (FP8 + BF8 + i4 + MX)
BlockScaleGemmSpec, build_block_scale_gemm # CK Tile 38
block_scale_gemm_grid, block_scale_gemm_signature
QuantMode, MantissaDType # enums
MxGemmSpec, MxMantissaDType, build_mx_gemm # CK Tile 42
mx_gemm_grid, mx_gemm_signature

# fused MoE forward (gather + SwiGLU + topk-weighted reduce)
FusedMoeSpec, FusedMoeLauncher # CK Tile 15
build_moe_gather, moe_gather_grid, moe_gather_signature
build_moe_silu_mul, moe_silu_mul_grid, moe_silu_mul_signature
build_moe_topk_weighted_reduce, moe_topk_weighted_reduce_grid
moe_topk_weighted_reduce_signature
moe_fused_workspace_bytes

# FMHA expansion (CK Tile 01_fmha family)
# Shared scaffolding
FmhaCommonSpec, FmhaShape, FmhaMaskMode
validate_fmha_common_spec

# Variants (each ships build_*, *_grid, *_signature, is_valid_*_spec)
FmhaFwdVarlenSpec, build_fmha_fwd_varlen # varlen
FmhaAppendKvSpec, build_fmha_fwd_appendkv # appendkv
FmhaFwdPagedPrefillSpec, build_fmha_fwd_paged_prefill # paged-prefill
FmhaFwdSplitKvDecodeSpec # split-kv decode (2 launches)
build_fmha_fwd_splitkv_decode_segment, build_fmha_fwd_splitkv_decode_reduce
FmhaFwdHeadGroupingSpec, build_fmha_fwd_head_grouping # GQA / MQA
FmhaBwdSpec, build_fmha_bwd # dQ / dK / dV
FmhaFwdFp8Spec, build_fmha_fwd_fp8 # fp8 KV cache

# Sage attention (4 quant_mode variants) + sparse attention
SageAttentionSpec, SageQuantMode # CK Tile 49_sageattention
build_sage_attention, sage_attention_grid, sage_attention_signature
JengaSparseSpec, build_jenga_sparse_attention # CK Tile 50_sparse_attn jenga
jenga_sparse_attention_grid, jenga_sparse_attention_signature
VsaSparseSpec, build_vsa_sparse_attention # CK Tile 50_sparse_attn vsa
vsa_sparse_attention_grid, vsa_sparse_attention_signature
```

## New IR / IRBuilder ops added in

```text
# : quantisation casts (single-element)
b.cvt_bf8_to_f32(v) # bf8e5m2 -> f32 (llvm.amdgcn.cvt.f32.bf8)
b.cvt_f32_to_fp8(v) # f32 -> fp8e4m3 (llvm.amdgcn.cvt.pk.fp8.f32)
b.cvt_f32_to_bf8(v) # f32 -> bf8e5m2 (llvm.amdgcn.cvt.pk.bf8.f32)
b.cvt_f32_to_i8_sat(v) # f32 -> i8 (rint + smin/smax clamp + trunc)
b.clamp_f32(v, lo, hi) # fmin(hi, fmax(lo, v)) -- folds to v_med3_f32

# : atomics (used by MoE-sort, StreamK-GEMM,
# and FMHA-backward dQ accumulate)
b.global_atomic_add(ptr, idx, val, *, ordering="monotonic")
b.lds_atomic_add(smem, [idx], val, *, ordering="monotonic")

# : bitwise integer helpers (also used by i4 unpack)
b.xor(a, b) # arith.xor; same type on both operands
b.shl(a, b) # arith.shl; logical left shift

# : extra integer ops + packed bf16 atomic (for FMHA / RNG)
b.lshr(a, b) # arith.lshr; logical right shift
b.umul_hi_i32(a, b) # high 32 bits of u32*u32 (Philox mulhilo32)
b.global_atomic_add_pk_bf16(ptr, idx, vec2_bf16)
 # llvm.amdgcn.global.atomic.fadd.v2bf16 -- packed
 # bf16 atomic for FMHA-bwd dQ direct-bf16 path

# : FP8 / BF8 MFMA (operands are <8 x fp8e4m3> / <8 x bf8e5m2>;
# lowering bitcasts to <2 x i32> for the amdgcn intrinsic)
b.mfma_f32_16x16x32_fp8(a, b, c) # llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8
b.mfma_f32_16x16x32_bf8(a, b, c) # llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8
b.mfma_f32_32x32x16_fp8(a, b, c) # llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8
b.mfma_f32_32x32x16_bf8(a, b, c) # llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8
```

## launch helper

The persistent-kernel + StreamK partitioner pair is the canonical
deliverable shared between the MoE pipeline (/ 6) and the
StreamK GEMM family (/ 5). Usage::

 from ck_dsl.helpers import (
 persistent_tile_for_each,
 emit_streamk_decode,
 StreamKPartition)

 partition = StreamKPartition(m_tiles=16, n_tiles=16, k_iters=8)

 def body(linear_id):
 decoded = emit_streamk_decode(b, linear_id, partition)
 # ... per-macro-tile work using
 # (decoded.m_tile, decoded.n_tile, decoded.k_iter,
 # decoded.is_first, decoded.is_last)

 persistent_tile_for_each(
 b,
 counter=Counter, # i32 global, pre-cleared
 num_tiles=b.const_i32(partition.num_macro_tiles),
 max_iters=ceil(partition.num_macro_tiles / launch_grid_size),
 body=body)

See ``instances/streamk_gemm.py`` for the minimal end-to-end example.

## `from ck_dsl.transforms import ...`

```text
CoordVar, Transform
PassThrough, Pad, PadDynamic, Embed, Merge, Unmerge, Indirect
TensorDescriptor
pass_through, pad, pad_dynamic, embed, merge, unmerge, indirect
```

## Submodule entry points

```text
python -m ck_dsl # list discoverable entry points
python -m ck_dsl.run_manifest # numpy + HIP manifest runner
python -m ck_dsl.examples.bake_off_implicit_gemm # example generator
python -m ck_dsl.sweep_bench # benchmark a sweep manifest
```
