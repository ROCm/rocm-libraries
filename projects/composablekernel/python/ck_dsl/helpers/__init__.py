# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""High-level helpers for authoring CK Tile-style GPU kernels.

This package sits one layer above `ck_dsl._ir` (the SSA IR + builder) and
captures the patterns that every CK Tile-style GEMM, attention, or
convolution kernel re-implements by hand. Two cohesive layers:

CK Tile-inspired data abstractions (port of ``make_tensor_view`` etc.)
  - `TensorDescriptor`        - shape + strides + dtype (``tensor_descriptor``).
                                Stride entries can be ``int`` (compile-time)
                                or SSA ``Value`` (runtime), paralleling
                                CK Tile's ``number<>`` vs ``index_t``
                                distinction.
  - `TensorView`              - pointer + descriptor + addr space
                                (``tensor_view<addr_space::*>``)
  - `TileWindow`              - moveable origin + extents into a TensorView
                                (``tile_window``). Supports ``move_to`` /
                                ``shift_by`` for sliding-window patterns.
  - `TensorCoordinate`,         Cached (index, offset) pair with incremental
    `make_tensor_coordinate`,   ``move`` updates (``tensor_coordinate`` +
    `move_tensor_coordinate`    ``move_tensor_coordinate``).
  - `make_global_view`,         ``make_tensor_view<addr_space::global, lds>``
    `make_lds_view`             plus ``make_naive_tensor_descriptor_packed``.
  - `make_naive_tensor_view_packed`,  CK Tile literal-name aliases for the
    `make_tile_window`                two free-function forms; use either.
  - `view_from_transforms_descriptor`  Bridge ``ck_dsl.transforms`` (rich
                                       transform-pipeline descriptors with
                                       named coords) into the :class:`TensorView`
                                       API; discards validity masks for now.
  - `sweep_row_chunks`,         CK Tile-style "load X once, sweep Y" iteration
    `pass2_row_chunks`          helpers; see ``ck_tile/core/tensor/sweep_tile.hpp``
                                and :ref:`ck_tile_sweep_tile`.
  - `TileDistributionEncoding`,  Full CK Tile distribution machinery (v1: no R,
    `make_static_tile_distribution`,  1D-2D X, Ps/Ys flexible). The encoding
    `TileDistribution`,               carries the (Rs, Hs, Ps2RHs, Ys2RHs)
    `LoadStoreTraits`,                tuples; :func:`make_load_store_traits`
    `make_load_store_traits`,         analyses it to pick `vector_dim_y` +
    `StaticDistributedTensor`,        `scalar_per_vector` + snake traversal
    `load_tile`, `store_tile`         order; :func:`load_tile` / `store_tile`
                                      drive an automated, vectorised
                                      window <-> register-tile pass.
  - `io_ir_type`, `load_vec`, `store_vec`, `load_vec_as_f32`,
    `pack_f32_to`             - dtype-string-tolerant I/O dispatch
  - `block_lds_reduce`        - canonical LDS tree reduction (sum / max)
  - `IOSpecRule`, `validate_io`,
    `kernel_name_join`,
    `SignatureBuilder`,
    `ceil_div_grid`           - one-line spec / signature / grid helpers

Kernel-shape abstractions (GEMM / conv / attention infrastructure):
  - `MfmaAtom`                - one matrix-multiply intrinsic + its lane layout
  - `WarpGrid`                - block/warp/lane decomposition and constants
  - `CoalescedTileLoader`     - coalesced global-to-LDS sync tile copy
  - `AsyncTileLoader`         - direct global-to-LDS async DMA copy (compv4)
  - `DirectEpilogue`          - per-lane vector global stores
  - `CShuffleEpilogue`        - LDS-staged wide-vector global stores
  - `compile_kernel`          - one-shot IR -> LLVM IR -> HSACO pipeline
  - `make_gemm_manifest`      - the standard manifest.json schema
  - `make_conv_manifest`      - manifest schema for convolution kernels

The IR primitives in `ck_dsl._ir` remain the source of truth and the
escape hatch: anything the helpers do can be re-derived directly from
the IR builder when a kernel needs something custom. The helpers exist
because *every* GEMM kernel in this repo (dsl/01 through dsl/07) and
*every* convolution kernel (dsl/08+) builds the same five-or-six-step
skeleton, and writing it from scratch is the single biggest source of
bugs (lane-mapping errors, off-by-one in the load distribution, missing
barriers, wrong epilogue indexing).

See `python/ck_dsl/helpers/README.md` for a top-to-bottom worked
example that uses every helper in this module, and
`python/ck_dsl/TRANSFORM_DAG.md` for how the coord-transform algebra
in `ck_dsl.transforms` composes with these helpers to build full
convolution kernels in the CK Tile style.
"""

from __future__ import annotations

from .atoms import MFMA_F16_ATOMS, MfmaAtom, mfma_atom
from .autotune import (
    AutotuneConfig,
    AutotuneKey,
    AutotuneResult,
    Autotuner,
    autotune_sweep,
    make_autotune_key,
    spec_replace,
)
from .attention import (
    Attention2DConfig,
    Attention3DConfig,
    OnlineSoftmaxState,
    PagedKvDescriptor,
    apply_softcap_log2,
    apply_softcap_scalar,
    binary_search_seq_idx,
    causal_mask,
    mfma_16x16x16_for_dtype,
    mfma_16x16x32_for_dtype,
    select_2d_config,
    select_3d_config,
    sliding_window_mask,
    use_2d_kernel,
    warp_xor_reduce_max,
    warp_xor_reduce_sum,
)
from .compile import KernelArtifact, compile_kernel
from .distribution import (
    LoadStoreTraits,
    StaticDistributedTensor,
    TileDistribution,
    TileDistributionEncoding,
    load_tile,
    make_load_store_traits,
    make_static_distributed_tensor,
    make_static_tile_distribution,
    store_tile,
)
from .epilogues import CShuffleEpilogue, DirectEpilogue
from .fuse import (
    BiasAdd,
    Cast,
    Clamp,
    EpilogueOp,
    FusedEpilogue,
    FusionMatchError,
    FusionPlan,
    GELU,
    ReLU,
    ResidualAdd,
    ResidualMul,
    Scale,
    SiLU,
    compile_fn,
    dtype_to_ir,
    explain_fn,
    fuse_matmul_bias_relu,
    ir_dtype_const,
    ir_dtype_global_load,
    ir_dtype_zero,
)
from .fusion_ir import (
    FusionGraph,
    FusionOp,
    FusionRegion,
    FusionTensor,
    build_graph,
)
from .fusion_legalize import FusionLegalizer, LegalResult
from .fusion_lowering import (
    BuiltRegion,
    ElementwiseLowerer,
    ExplainOnlyLowerer,
    GemmEpilogueLowerer,
    LoweringRegistry,
    ReductionLowerer,
    default_lowering_registry,
)
from .fusion_memory import (
    WorkspaceAllocation,
    WorkspacePlanner,
    materialize_plan,
)
from .fusion_scheduler import GreedyFusionScheduler, RegionCost
from .fusion_validation import (
    BackendTiming,
    BenchmarkCase,
    FusionMatrixRunner,
    ValidationReport,
    run_fusion_validation_matrix,
)
from .geometry import WarpGrid
from .grid import (
    NUM_XCDS_MI300X,
    NUM_XCDS_MI325X,
    NUM_XCDS_MI350X,
    SuperTileSwizzleResult,
    chiplet_aware_super_tile,
    chiplet_aware_super_tile_dynamic,
    chiplet_transform_chunked,
    chiplet_transform_chunked_dynamic,
    python_chiplet_transform_chunked,
    python_super_tile_swizzle,
    super_tile_swizzle,
    super_tile_swizzle_dynamic,
)
from .io import (
    io_ir_type,
    load_scalar,
    load_scalar_as_f32,
    load_vec,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar,
    store_scalar_from_f32,
    store_vec,
)
from .layouts import LdsLayout, TransposeLdsReader
from .loads import (
    AsyncTileLoader,
    AsyncTileLoaderSlot,
    CoalescedTileLoader,
    DescriptorFn,
    lane_contiguous_descriptor,
)
from .pipeline import SoftwarePipeline
from .reduction import ReduceCombine, block_lds_reduce
from .schedule import SchedulePolicy
from .spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    ptr_type_str,
    sig_param,
    sig_scalar,
    validate_io,
)
from .sweep import (
    RowChunkSweepResult,
    pass2_row_chunks,
    sweep_row_chunks,
)
from .tensor_view import (
    BufferResource,
    TensorCoordinate,
    TensorDescriptor,
    TensorView,
    TileWindow,
    make_buffer_resource,
    make_buffer_view,
    make_global_view,
    make_lds_view,
    make_naive_tensor_descriptor_packed,
    make_naive_tensor_view_packed,
    make_tensor_coordinate,
    make_tile_window,
    move_tensor_coordinate,
    view_from_transforms_descriptor,
)
from .manifest import (
    MANIFEST_SCHEMA,
    attention_args_signature,
    conv_args_signature,
    gemm_args_signature,
    make_attention_manifest,
    make_conv_manifest,
    make_gemm_manifest,
    make_simple_op_manifest,
    write_artifact,
)

__all__ = [
    # Atoms
    "Attention2DConfig",
    "Attention3DConfig",
    "MFMA_F16_ATOMS",
    "MfmaAtom",
    "OnlineSoftmaxState",
    "PagedKvDescriptor",
    "apply_softcap_log2",
    "apply_softcap_scalar",
    "binary_search_seq_idx",
    "causal_mask",
    "mfma_16x16x16_for_dtype",
    "mfma_16x16x32_for_dtype",
    "mfma_atom",
    "select_2d_config",
    "select_3d_config",
    "sliding_window_mask",
    "use_2d_kernel",
    "warp_xor_reduce_max",
    "warp_xor_reduce_sum",
    # Geometry
    "WarpGrid",
    # Autotuning
    "AutotuneConfig",
    "AutotuneKey",
    "AutotuneResult",
    "Autotuner",
    "autotune_sweep",
    "make_autotune_key",
    "spec_replace",
    # Chiplet / grid swizzle (multi-XCD L2 locality)
    "NUM_XCDS_MI300X",
    "NUM_XCDS_MI325X",
    "NUM_XCDS_MI350X",
    "SuperTileSwizzleResult",
    "chiplet_aware_super_tile",
    "chiplet_aware_super_tile_dynamic",
    "chiplet_transform_chunked",
    "chiplet_transform_chunked_dynamic",
    "python_chiplet_transform_chunked",
    "python_super_tile_swizzle",
    "super_tile_swizzle",
    "super_tile_swizzle_dynamic",
    # Loads
    "LdsLayout",
    "TransposeLdsReader",
    "AsyncTileLoader",
    "AsyncTileLoaderSlot",
    "CoalescedTileLoader",
    "DescriptorFn",
    "lane_contiguous_descriptor",
    "SchedulePolicy",
    "SoftwarePipeline",
    # Epilogues
    "CShuffleEpilogue",
    "DirectEpilogue",
    # Fusion (graph capture + pattern match + lower)
    "BiasAdd",
    "Cast",
    "Clamp",
    "EpilogueOp",
    "FusedEpilogue",
    "FusionMatchError",
    "FusionPlan",
    "GELU",
    "ReLU",
    "ResidualAdd",
    "ResidualMul",
    "Scale",
    "SiLU",
    "compile_fn",
    "dtype_to_ir",
    "explain_fn",
    "fuse_matmul_bias_relu",
    "ir_dtype_const",
    "ir_dtype_global_load",
    "ir_dtype_zero",
    # Fusion IR / solver scaffolding
    "FusionGraph",
    "FusionOp",
    "FusionRegion",
    "FusionTensor",
    "build_graph",
    "FusionLegalizer",
    "LegalResult",
    "BuiltRegion",
    "ElementwiseLowerer",
    "ExplainOnlyLowerer",
    "GemmEpilogueLowerer",
    "LoweringRegistry",
    "ReductionLowerer",
    "default_lowering_registry",
    "WorkspaceAllocation",
    "WorkspacePlanner",
    "materialize_plan",
    "GreedyFusionScheduler",
    "RegionCost",
    "BackendTiming",
    "BenchmarkCase",
    "FusionMatrixRunner",
    "ValidationReport",
    "run_fusion_validation_matrix",
    # Compile
    "KernelArtifact",
    "compile_kernel",
    # Manifest
    "MANIFEST_SCHEMA",
    "attention_args_signature",
    "conv_args_signature",
    "gemm_args_signature",
    "make_attention_manifest",
    "make_conv_manifest",
    "make_gemm_manifest",
    "make_simple_op_manifest",
    "write_artifact",
    # CK Tile-inspired tensor abstractions
    "BufferResource",
    "TensorCoordinate",
    "TensorDescriptor",
    "TensorView",
    "TileWindow",
    "make_buffer_resource",
    "make_buffer_view",
    "make_global_view",
    "make_lds_view",
    "make_naive_tensor_descriptor_packed",
    "make_naive_tensor_view_packed",
    "make_tensor_coordinate",
    "make_tile_window",
    "move_tensor_coordinate",
    "view_from_transforms_descriptor",
    # Distribution + distributed tensor + load_store_traits
    "LoadStoreTraits",
    "StaticDistributedTensor",
    "TileDistribution",
    "TileDistributionEncoding",
    "load_tile",
    "make_load_store_traits",
    "make_static_distributed_tensor",
    "make_static_tile_distribution",
    "store_tile",
    # Sweep iteration
    "RowChunkSweepResult",
    "pass2_row_chunks",
    "sweep_row_chunks",
    # I/O dispatch
    "io_ir_type",
    "load_scalar",
    "load_scalar_as_f32",
    "load_vec",
    "load_vec_as_f32",
    "pack_f32_to",
    "store_scalar",
    "store_scalar_from_f32",
    "store_vec",
    # Reductions
    "ReduceCombine",
    "block_lds_reduce",
    # Spec / signature / grid
    "IOSpecRule",
    "SignatureBuilder",
    "ceil_div_grid",
    "kernel_name_join",
    "ptr_type_str",
    "sig_param",
    "sig_scalar",
    "validate_io",
]
