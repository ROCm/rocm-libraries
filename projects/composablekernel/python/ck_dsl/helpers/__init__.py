# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""High-level helpers for authoring CK Tile-style GPU kernels.

This package sits one layer above `ck_dsl._ir` (the SSA IR + builder) and
captures the patterns that every CK Tile-style GEMM, attention, or
convolution kernel re-implements by hand:

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
from .attention import (
    Attention2DConfig,
    Attention3DConfig,
    OnlineSoftmaxState,
    PagedKvDescriptor,
    apply_softcap_scalar,
    causal_mask,
    select_2d_config,
    select_3d_config,
    sliding_window_mask,
    use_2d_kernel,
)
from .compile import KernelArtifact, compile_kernel
from .epilogues import CShuffleEpilogue, DirectEpilogue
from .geometry import WarpGrid
from .layouts import LdsLayout
from .loads import (
    AsyncTileLoader,
    AsyncTileLoaderSlot,
    CoalescedTileLoader,
    DescriptorFn,
    lane_contiguous_descriptor,
)
from .pipeline import SoftwarePipeline
from .schedule import SchedulePolicy
from .manifest import (
    MANIFEST_SCHEMA,
    attention_args_signature,
    conv_args_signature,
    gemm_args_signature,
    make_attention_manifest,
    make_conv_manifest,
    make_gemm_manifest,
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
    "apply_softcap_scalar",
    "causal_mask",
    "mfma_atom",
    "select_2d_config",
    "select_3d_config",
    "sliding_window_mask",
    "use_2d_kernel",
    # Geometry
    "WarpGrid",
    # Loads
    "LdsLayout",
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
    "write_artifact",
]
