// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared argument structs and compile-time configuration for the FMHA BWD
// OGradDotO kernel family.
//
// This header has no CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI and
// compile-time-validated configuration. Requires HIP for dim3.

#pragma once

#include <rocm_ck/datatype_utils.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// FMHA attention mode: fixed-length batches vs variable-length groups.
enum class FmhaMode
{
    BATCH,
    GROUP
};

// ---------------------------------------------------------------------------
// Signature / Algorithm / Config / Kernel
// ---------------------------------------------------------------------------

/// Signature: describes WHAT the kernel computes (data types, dimensions, mode).
struct FmhaBwdOGradDotOSignature
{
    DataType dtype; // fp16 or bf16 (controls O, dO types; D is always float)
    int hdim_v;     // head dimension: 32, 64, 96, 128, 256
    FmhaMode mode;  // batch or group
};

/// Algorithm: describes HOW the kernel executes (padding, occupancy, block size).
struct FmhaBwdOGradDotOAlgorithm
{
    bool pad_seqlen_q;     // kPadSeqLenQ
    bool pad_hdim_v;       // kPadHeadDimV
    int block_per_cu = 2;  // occupancy hint (default from TileFmhaBwdOGradDotOTraits)
    int block_size   = 64; // tile_m0 = kM0 = kBlockSize
};

/// Config: user-facing Signature + Algorithm pair.
struct FmhaBwdOGradDotOConfig
{
    FmhaBwdOGradDotOSignature signature;
    FmhaBwdOGradDotOAlgorithm algorithm;
};

/// Validated kernel descriptor — structural type, safe for use as NTTP.
/// All optional/default values are resolved; no std::optional.
struct FmhaBwdOGradDotOKernel
{
    DataType dtype;
    int hdim_v;
    FmhaMode mode;
    bool pad_seqlen_q;
    bool pad_hdim_v;
    int block_per_cu;
    int block_size;
};

/// Validate config and produce a structural kernel descriptor.
consteval FmhaBwdOGradDotOKernel make_kernel(FmhaBwdOGradDotOConfig cfg)
{
    auto sig  = cfg.signature;
    auto algo = cfg.algorithm;

    if(sig.dtype != DataType::FP16 && sig.dtype != DataType::BF16)
        throw "FmhaBwdOGradDotO only supports FP16 or BF16 (O/dO types; D is always float)";

    // Valid head dimensions per BWD_DOT_DO_O_HDIMS in dispatcher
    if(sig.hdim_v != 32 && sig.hdim_v != 64 && sig.hdim_v != 96 && sig.hdim_v != 128 &&
       sig.hdim_v != 256)
        throw "hdim_v must be one of {32, 64, 96, 128, 256}";

    // BlockFmhaBwdOGradDotOPipelineProblem requires:
    //   0 < kBlockSize && kBlockSize % get_warp_size() == 0
    if(algo.block_size <= 0)
        throw "block_size must be positive";
    if(algo.block_size % 64 != 0)
        throw "block_size must be divisible by warp size (64)";

    if(algo.block_per_cu <= 0)
        throw "block_per_cu must be positive";

    // Group mode requires seqlen padding (variable-length sequences)
    if(sig.mode == FmhaMode::GROUP && !algo.pad_seqlen_q)
        throw "group mode requires pad_seqlen_q=true (variable-length sequences)";

    return {sig.dtype,
            sig.hdim_v,
            sig.mode,
            algo.pad_seqlen_q,
            algo.pad_hdim_v,
            algo.block_per_cu,
            algo.block_size};
}

// --- make_kernel compile-time tests ---
// clang-format off

// Valid configs compile:
static_assert(make_kernel(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}}).dtype == DataType::FP16);
static_assert(make_kernel(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::BF16, .hdim_v = 64, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}}).hdim_v == 64);
static_assert(make_kernel(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}}).mode == FmhaMode::GROUP);
static_assert(make_kernel(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}}).pad_seqlen_q == false);

// Invalid configs (uncommenting produces consteval compile errors):
// make_kernel({.signature = {.dtype = DataType::FP32, ...}})  — FP32 not supported
// make_kernel({.signature = {.hdim_v = 100, ...}})            — invalid hdim
// make_kernel({.signature = {.mode = FmhaMode::GROUP}, .algorithm = {.pad_seqlen_q = false}})
//   — group mode requires pad_seqlen_q

// clang-format on

// ---------------------------------------------------------------------------
// Kernel arguments — flat structs matching CK Tile's internal Kargs layout
// ---------------------------------------------------------------------------

/// Batch-mode kernel arguments.
/// Layout matches CK Tile's FmhaBwdOGradDotOBatchModeKargs exactly.
struct FmhaBwdOGradDotOBatchArgs
{
    // --- Common (matches FmhaBwdOGradDotOCommonKargs) ---
    const void* o_ptr;
    const void* do_ptr;
    void* d_ptr;

    float p_undrop;

    index_t seqlen_q;
    index_t hdim_v;

    index_t stride_do;
    index_t stride_o;

    index_t nhead_stride_do;
    index_t nhead_stride_o;
    index_t nhead_stride_d;

    // --- Batch extension ---
    index_t batch_stride_do;
    index_t batch_stride_o;
    index_t batch_stride_d;
};

static_assert(std::is_trivially_copyable_v<FmhaBwdOGradDotOBatchArgs>,
              "FmhaBwdOGradDotOBatchArgs must be trivially copyable for kernarg passing");
static_assert(std::is_standard_layout_v<FmhaBwdOGradDotOBatchArgs>,
              "FmhaBwdOGradDotOBatchArgs must be standard layout");

/// Group-mode kernel arguments.
/// Layout matches CK Tile's FmhaBwdOGradDotOGroupModeKargs exactly.
struct FmhaBwdOGradDotOGroupArgs
{
    // --- Common (same fields as batch, seqlen_q = -1 placeholder) ---
    const void* o_ptr;
    const void* do_ptr;
    void* d_ptr;

    float p_undrop;

    index_t seqlen_q; // -1 (updated per-batch on device)
    index_t hdim_v;

    index_t stride_do;
    index_t stride_o;

    index_t nhead_stride_do;
    index_t nhead_stride_o;
    index_t nhead_stride_d;

    // --- Group extension ---
    const int32_t* seqstart_q_ptr;
    const int32_t* seqlen_q_ptr;
    const int32_t* cu_seqlen_q_ptr;
};

static_assert(std::is_trivially_copyable_v<FmhaBwdOGradDotOGroupArgs>,
              "FmhaBwdOGradDotOGroupArgs must be trivially copyable for kernarg passing");
static_assert(std::is_standard_layout_v<FmhaBwdOGradDotOGroupArgs>,
              "FmhaBwdOGradDotOGroupArgs must be standard layout");

// ---------------------------------------------------------------------------
// Grid calculation
// ---------------------------------------------------------------------------

/// Compute the launch grid for OGradDotO.
/// Matches FmhaBwdOGradDotOKernel::GridSize(): dim3(ceil(seqlen_q / kM0), nhead, batch).
constexpr dim3 grid_size(int batch, int nhead, int seqlen_q, int block_size)
{
    return dim3(static_cast<unsigned>((seqlen_q + block_size - 1) / block_size),
                static_cast<unsigned>(nhead),
                static_cast<unsigned>(batch));
}

} // namespace rocm_ck
