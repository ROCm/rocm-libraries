// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv {

/// Map lanes to row, col, and batch for transposed loading of 16-bit matrices from LDS.
/// For use with DS_READ_B64_TR_B16 on CDNA4.
///
/// Template parameters match MatrixLayout<M, K, B>:
///   M = outer-product dimension (columns of the source matrix)
///   K = inner-product dimension (rows of the source matrix)
///   B = batch count
///
/// Each ds_read_b64_tr_b16 call reads 4 consecutive column elements.
/// READS = number of calls needed per lane (K_L / 4).
///
/// Address for read i:  base + row(lane, i) * row_stride + col(lane)
template <int M, int K, int B = 1>
struct TransposeLDSLayout
{
    static_assert(M > 0 && K > 0 && B > 0, "M, K, B must be positive.");
    static_assert(M * B <= 64, "M * B must not exceed wavefront size (64).");
    static_assert(M % 4 == 0, "M must be divisible by 4 (ds_read_b64_tr_b16 reads 4 elements).");
    static_assert(K % (64 / (M * B)) == 0, "K must be divisible by 64 / (M * B).");

    static constexpr int K_L = K / (64 / (M * B));
    static_assert(K_L % 4 == 0, "K_L must be divisible by 4 (items per ds_read_b64_tr_b16).");

    static constexpr int READS = K_L / 4;

    /// The row of the source matrix this lane reads from for the i-th ds_read call.
    __host__ __device__ static constexpr int row(int lane, int read_idx = 0)
    { return (lane / (M * B)) * K_L + read_idx * 4 + (lane / 4) % 4; }

    /// The starting column of the source matrix this lane reads from.
    /// The lane reads 4 consecutive elements: col, col+1, col+2, col+3.
    __host__ __device__ static constexpr int col(int lane) { return ((lane % 4) % (M / 4)) * 4; }

    /// The batch index for this lane (only meaningful when B > 1).
    __host__ __device__ static constexpr int batch(int lane)
    { return ((lane / 16) * (16 / M) + (lane % 4) / (M / 4)) % B; }
};

} // namespace ck_tile::direct_conv
