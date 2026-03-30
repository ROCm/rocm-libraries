// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: abi — shared between host and device. Trivially copyable, no CK deps.
//
// Generic kernel argument structure for all rocm_ck operations.
//
// One Args type replaces per-operation structs (VectorAddArgs, GemmArgs, etc.).
// Slot ordering matches the Signature: Args::tensors[i] <-> Signature::tensors[i].
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI.

#pragma once

#include <rocm_ck/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocm_ck {

/// Maximum tensor rank. Covers grouped 3D convolution layouts (GNCDHW = rank 6).
constexpr int kMaxRank = 6;

/// Maximum number of tensor slots. FMHA backward uses ~12 tensors.
constexpr int kMaxTensors = 16;

/// Maximum number of scalar slots. FMHA with masking+dropout needs ~12 scalars.
constexpr int kMaxScalars = 16;

/// A single tensor argument: pointer + shape + strides.
///
/// Unused dimensions have lengths[i] = 0, strides[i] = 0.
/// All pointers are const void* — output tensors use const_cast on the device side.
struct TensorArg
{
    const void* ptr;                       //  8 bytes  (offset 0)
    std::array<index_t, kMaxRank> lengths; // 24 bytes  (offset 8)   — int32
    std::array<int64_t, kMaxRank> strides; // 48 bytes  (offset 32)  — int64
};

/// A scalar value passed by value through kernarg.
///
/// No runtime type tag — the Signature declares each scalar's type at compile time.
/// The entry kernel knows which union member to read.
///
/// FP16/BF16/FP8 scalars use the f32 member with host-side conversion. Scalar
/// precision is always >= tensor precision, so float covers all sub-32-bit types.
union ScalarValue
{
    float f32;
    int32_t i32;
    uint32_t u32;
    double f64;
    int64_t i64;
    uint64_t u64;
};

/// Generic kernel arguments for all rocm_ck operations.
///
/// Slot ordering matches Signature: tensors[i] <-> Signature::tensors[i].
/// Trivially copyable, standard layout — required for kernarg passing.
///
/// sizeof = 1552 bytes (38% of the 4096-byte HSA kernarg budget).
struct Args
{
    std::array<TensorArg, kMaxTensors> tensors;   // 16 x 80 = 1280 bytes
    std::array<ScalarValue, kMaxScalars> scalars; // 16 x  8 =  128 bytes

    // Batch parameters (0 = unbatched, >0 = batched GEMM with blockIdx.y indexing)
    index_t batch_count = 0; //  4 bytes
    // Per-tensor batch stride in elements (0 = broadcast across batch)
    std::array<int64_t, kMaxTensors> batch_strides = {}; // 16 x 8 = 128 bytes

    // Workspace pointer for Stream-K partial reduction (nullptr when unused)
    void* workspace_ptr = nullptr; //  8 bytes
};

// =============================================================================
// Shape / stride helpers — zero-fill unused dimensions
// =============================================================================

/// Build a shape array from 1–6 dimensions, zero-filling unused slots.
/// Eliminates trailing-zero noise in example code:
///   make_shape(M, K) instead of {M, K, 0, 0, 0, 0}
constexpr std::array<index_t, kMaxRank> make_shape(
    index_t d0, index_t d1 = 0, index_t d2 = 0, index_t d3 = 0, index_t d4 = 0, index_t d5 = 0)
{
    return {d0, d1, d2, d3, d4, d5};
}

/// Build a strides array from 1–6 dimensions, zero-filling unused slots.
/// Eliminates trailing-zero noise in example code:
///   make_strides(stride_A, 1) instead of {stride_A, 1, 0, 0, 0, 0}
constexpr std::array<int64_t, kMaxRank> make_strides(
    int64_t s0, int64_t s1 = 0, int64_t s2 = 0, int64_t s3 = 0, int64_t s4 = 0, int64_t s5 = 0)
{
    return {s0, s1, s2, s3, s4, s5};
}

// =============================================================================
// ABI stability assertions
// =============================================================================

// --- TensorArg ---
static_assert(std::is_trivially_copyable_v<TensorArg>,
              "TensorArg must be trivially copyable for kernarg passing");
static_assert(std::is_standard_layout_v<TensorArg>,
              "TensorArg must be standard layout for kernarg passing");
static_assert(sizeof(TensorArg) == 80, "unexpected TensorArg size");
static_assert(alignof(TensorArg) == 8, "unexpected TensorArg alignment");
static_assert(offsetof(TensorArg, ptr) == 0, "unexpected offset for ptr");
static_assert(offsetof(TensorArg, lengths) == 8, "unexpected offset for lengths");
static_assert(offsetof(TensorArg, strides) == 32, "unexpected offset for strides");

// --- ScalarValue ---
static_assert(std::is_trivially_copyable_v<ScalarValue>,
              "ScalarValue must be trivially copyable for kernarg passing");
static_assert(sizeof(ScalarValue) == 8, "unexpected ScalarValue size");

// --- Args ---
static_assert(std::is_trivially_copyable_v<Args>,
              "Args must be trivially copyable for kernarg passing");
static_assert(std::is_standard_layout_v<Args>, "Args must be standard layout for kernarg passing");
static_assert(sizeof(Args) == 1552, "unexpected Args size");
static_assert(alignof(Args) == 8, "unexpected Args alignment");

} // namespace rocm_ck
