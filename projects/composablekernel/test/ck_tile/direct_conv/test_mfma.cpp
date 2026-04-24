// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/utils/types.hpp"
#include "ck_tile/ops/direct_convolution/utils/mfma.hpp"
#pragma clang diagnostic pop

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace ck_tile::direct_conv;

// ============================================================================
// GPU kernel wrappers
// ============================================================================

// Each thread reads its fp16x4 A and B from device memory (indexed by lane ID),
// calls the MFMA functor, and writes the fp32x4 result to device memory.

__global__ void test_mfma_4x4x4_kernel(const fp16x4_t* __restrict__ a,
                                        const fp16x4_t* __restrict__ b,
                                        fp32x4_t* __restrict__ c)
{
    const int lane = threadIdx.x;
    Mfma4x4x4 mfma;
    fp32x4_t acc = {0.f, 0.f, 0.f, 0.f};
    c[lane] = mfma(a[lane], b[lane], acc);
}

__global__ void test_mfma_16x16x16_kernel(const fp16x4_t* __restrict__ a,
                                           const fp16x4_t* __restrict__ b,
                                           fp32x4_t* __restrict__ c)
{
    const int lane = threadIdx.x;
    Mfma16x16x16 mfma;
    fp32x4_t acc = {0.f, 0.f, 0.f, 0.f};
    c[lane] = mfma(a[lane], b[lane], acc);
}

// ============================================================================
// CPU reference implementations
// ============================================================================

// mfma_f32_4x4x4f16 on a 64-lane wave:
//   16 independent 4x4 matmuls (batches), one per group of 4 lanes.
//   lane_col   = lane % 4   -> column in the 4x4 output
//   lane_batch = lane / 4   -> batch index (0..15)
//
//   A operand for batch b: rows 0..3, each 4 fp16 from lane (b*4 + row).
//   B operand for batch b: cols 0..3, each 4 fp16 from lane (b*4 + col).
//   C[row][col] = sum_k(A[row][k] * B[col][k])
//
//   Output per lane: c[lane] = fp32x4 = { C[0][col], C[1][col], C[2][col], C[3][col] }
//   where col = lane % 4.
static void reference_mfma_4x4x4(const std::vector<fp16x4_t>& a_host,
                                  const std::vector<fp16x4_t>& b_host,
                                  std::vector<fp32x4_t>& c_ref)
{
    // For each of the 16 batches, compute 4x4 matmul
    for(int batch = 0; batch < 16; batch++)
    {
        // Collect A matrix [4 rows][4 cols] from lanes batch*4 .. batch*4+3
        float A[4][4];
        float B[4][4];
        for(int row = 0; row < 4; row++)
        {
            int lane_idx = batch * 4 + row;
            for(int k = 0; k < 4; k++)
            {
                A[row][k] = static_cast<float>(a_host[lane_idx][k]);
                B[row][k] = static_cast<float>(b_host[lane_idx][k]);
            }
        }

        // C[row][col] = sum_k A[row][k] * B[col][k]
        float C[4][4];
        for(int row = 0; row < 4; row++)
        {
            for(int col = 0; col < 4; col++)
            {
                float sum = 0.f;
                for(int k = 0; k < 4; k++)
                    sum += A[row][k] * B[col][k];
                C[row][col] = sum;
            }
        }

        // Write to per-lane output: lane (batch*4 + col) gets { C[0][col], C[1][col], C[2][col], C[3][col] }
        for(int col = 0; col < 4; col++)
        {
            int lane_idx       = batch * 4 + col;
            c_ref[lane_idx][0] = C[0][col];
            c_ref[lane_idx][1] = C[1][col];
            c_ref[lane_idx][2] = C[2][col];
            c_ref[lane_idx][3] = C[3][col];
        }
    }
}

// mfma_f32_16x16x16f16 on a 64-lane wave:
//   Single 16x16 matmul with K=16 reduction.
//   lane_q  = lane % 16  -> output column (N dimension)
//   lane_c4 = lane / 16  -> K-reduction group (0..3), each group contributes 4 fp16
//
//   A operand: lane provides A[lane_q][lane_c4*4 .. lane_c4*4+3]
//   B operand: lane provides B[lane_q][lane_c4*4 .. lane_c4*4+3]
//
//   The instruction computes: C[16x16] += A[16x16] * B[16x16]
//   where A is the "input" matrix and B is the "weight/filter" matrix.
//
//   Output per lane: c[lane] = fp32x4 = { C[lane_c4*4+0][lane_q],
//                                          C[lane_c4*4+1][lane_q],
//                                          C[lane_c4*4+2][lane_q],
//                                          C[lane_c4*4+3][lane_q] }
static void reference_mfma_16x16x16(const std::vector<fp16x4_t>& a_host,
                                     const std::vector<fp16x4_t>& b_host,
                                     std::vector<fp32x4_t>& c_ref)
{
    // Reconstruct full A[16][16] and B[16][16] from per-lane operands.
    //
    // For src0 (A operand): lane L provides A[row=L%16][k=(L/16)*4+elem].
    //   A[row][k] = a_host[(k/4)*16 + row][k%4]
    //
    // For src1 (B operand): lane L provides B[k=(L/16)*4+elem][col=L%16].
    //   B[k][col] = b_host[(k/4)*16 + col][k%4]
    float A[16][16];
    float B[16][16];

    for(int row = 0; row < 16; row++)
    {
        for(int k = 0; k < 16; k++)
        {
            int lane_idx = (k / 4) * 16 + row;
            A[row][k]    = static_cast<float>(a_host[lane_idx][k % 4]);
        }
    }
    for(int k = 0; k < 16; k++)
    {
        for(int col = 0; col < 16; col++)
        {
            int lane_idx = (k / 4) * 16 + col;
            B[k][col]    = static_cast<float>(b_host[lane_idx][k % 4]);
        }
    }

    // D[m][n] = sum_k A[m][k] * B[k][n]
    float C[16][16] = {};
    for(int m = 0; m < 16; m++)
    {
        for(int n = 0; n < 16; n++)
        {
            float sum = 0.f;
            for(int k = 0; k < 16; k++)
                sum += A[m][k] * B[k][n];
            C[m][n] = sum;
        }
    }

    // Map output to per-lane registers.
    // lane_q = lane % 16, lane_c4 = lane / 16
    // c[lane] = { C[lane_c4*4+0][lane_q], C[lane_c4*4+1][lane_q],
    //             C[lane_c4*4+2][lane_q], C[lane_c4*4+3][lane_q] }
    for(int lane = 0; lane < 64; lane++)
    {
        int lane_q  = lane % 16;
        int lane_c4 = lane / 16;
        c_ref[lane][0] = C[lane_c4 * 4 + 0][lane_q];
        c_ref[lane][1] = C[lane_c4 * 4 + 1][lane_q];
        c_ref[lane][2] = C[lane_c4 * 4 + 2][lane_q];
        c_ref[lane][3] = C[lane_c4 * 4 + 3][lane_q];
    }
}

// ============================================================================
// Test fixture
// ============================================================================

class MfmaTest : public ::testing::Test
{
    protected:
    static constexpr int WAVE_SIZE = 64;

    // Fill fp16x4 vectors with small integer values for exact comparison.
    static void fill_small_values(std::vector<fp16x4_t>& data, int seed)
    {
        for(int i = 0; i < static_cast<int>(data.size()); i++)
        {
            for(int j = 0; j < 4; j++)
            {
                // Use small values (range -2..2) to keep fp16 matmul exact.
                float val    = static_cast<float>(((i * 4 + j + seed) % 5) - 2);
                data[i][j] = static_cast<_Float16>(val);
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(MfmaTest, Mfma4x4x4_SmallIntegers)
{
    std::vector<fp16x4_t> a_host(WAVE_SIZE);
    std::vector<fp16x4_t> b_host(WAVE_SIZE);
    std::vector<fp32x4_t> c_ref(WAVE_SIZE);

    fill_small_values(a_host, 0);
    fill_small_values(b_host, 7);
    reference_mfma_4x4x4(a_host, b_host, c_ref);

    // Allocate device memory
    fp16x4_t* d_a = nullptr;
    fp16x4_t* d_b = nullptr;
    fp32x4_t* d_c = nullptr;

    ck_tile::hip_check_error(
        hipMalloc(&d_a, WAVE_SIZE * sizeof(fp16x4_t)));
    ck_tile::hip_check_error(
        hipMalloc(&d_b, WAVE_SIZE * sizeof(fp16x4_t)));
    ck_tile::hip_check_error(
        hipMalloc(&d_c, WAVE_SIZE * sizeof(fp32x4_t)));

    ck_tile::hip_check_error(
        hipMemcpy(d_a, a_host.data(), WAVE_SIZE * sizeof(fp16x4_t), hipMemcpyHostToDevice));
    ck_tile::hip_check_error(
        hipMemcpy(d_b, b_host.data(), WAVE_SIZE * sizeof(fp16x4_t), hipMemcpyHostToDevice));

    test_mfma_4x4x4_kernel<<<1, WAVE_SIZE>>>(d_a, d_b, d_c);
    ck_tile::hip_check_error(hipDeviceSynchronize());

    std::vector<fp32x4_t> c_gpu(WAVE_SIZE);
    ck_tile::hip_check_error(
        hipMemcpy(c_gpu.data(), d_c, WAVE_SIZE * sizeof(fp32x4_t), hipMemcpyDeviceToHost));

    // Compare
    for(int lane = 0; lane < WAVE_SIZE; lane++)
    {
        for(int elem = 0; elem < 4; elem++)
        {
            EXPECT_NEAR(c_gpu[lane][elem], c_ref[lane][elem], 1e-3f)
                << "Mfma4x4x4 mismatch at lane=" << lane << " elem=" << elem;
        }
    }

    (void)hipFree(d_a);
    (void)hipFree(d_b);
    (void)hipFree(d_c);
}

TEST_F(MfmaTest, Mfma16x16x16_SmallIntegers)
{
    std::vector<fp16x4_t> a_host(WAVE_SIZE);
    std::vector<fp16x4_t> b_host(WAVE_SIZE);
    std::vector<fp32x4_t> c_ref(WAVE_SIZE);

    fill_small_values(a_host, 3);
    fill_small_values(b_host, 11);
    reference_mfma_16x16x16(a_host, b_host, c_ref);

    // Allocate device memory
    fp16x4_t* d_a = nullptr;
    fp16x4_t* d_b = nullptr;
    fp32x4_t* d_c = nullptr;

    ck_tile::hip_check_error(
        hipMalloc(&d_a, WAVE_SIZE * sizeof(fp16x4_t)));
    ck_tile::hip_check_error(
        hipMalloc(&d_b, WAVE_SIZE * sizeof(fp16x4_t)));
    ck_tile::hip_check_error(
        hipMalloc(&d_c, WAVE_SIZE * sizeof(fp32x4_t)));

    ck_tile::hip_check_error(
        hipMemcpy(d_a, a_host.data(), WAVE_SIZE * sizeof(fp16x4_t), hipMemcpyHostToDevice));
    ck_tile::hip_check_error(
        hipMemcpy(d_b, b_host.data(), WAVE_SIZE * sizeof(fp16x4_t), hipMemcpyHostToDevice));

    test_mfma_16x16x16_kernel<<<1, WAVE_SIZE>>>(d_a, d_b, d_c);
    ck_tile::hip_check_error(hipDeviceSynchronize());

    std::vector<fp32x4_t> c_gpu(WAVE_SIZE);
    ck_tile::hip_check_error(
        hipMemcpy(c_gpu.data(), d_c, WAVE_SIZE * sizeof(fp32x4_t), hipMemcpyDeviceToHost));

    // Compare
    for(int lane = 0; lane < WAVE_SIZE; lane++)
    {
        for(int elem = 0; elem < 4; elem++)
        {
            EXPECT_NEAR(c_gpu[lane][elem], c_ref[lane][elem], 1e-3f)
                << "Mfma16x16x16 mismatch at lane=" << lane << " elem=" << elem;
        }
    }

    (void)hipFree(d_a);
    (void)hipFree(d_b);
    (void)hipFree(d_c);
}

TEST_F(MfmaTest, Mfma4x4x4_Identity)
{
    // Test with identity-like pattern: A = identity, B = identity for batch 0
    std::vector<fp16x4_t> a_host(WAVE_SIZE, fp16x4_t{0, 0, 0, 0});
    std::vector<fp16x4_t> b_host(WAVE_SIZE, fp16x4_t{0, 0, 0, 0});
    std::vector<fp32x4_t> c_ref(WAVE_SIZE);

    // Set batch 0 (lanes 0-3) to identity: A[row][k] = (row==k) ? 1 : 0
    for(int row = 0; row < 4; row++)
    {
        for(int k = 0; k < 4; k++)
        {
            a_host[row][k] = (row == k) ? static_cast<_Float16>(1.0f) : static_cast<_Float16>(0.0f);
            b_host[row][k] = (row == k) ? static_cast<_Float16>(1.0f) : static_cast<_Float16>(0.0f);
        }
    }

    reference_mfma_4x4x4(a_host, b_host, c_ref);

    // For identity * identity, batch 0 output should be identity:
    // C[row][col] = (row == col) ? 1 : 0
    for(int col = 0; col < 4; col++)
    {
        int lane = col; // batch 0, lane_col = col
        for(int row = 0; row < 4; row++)
        {
            float expected = (row == col) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(c_ref[lane][row], expected)
                << "Identity test: lane=" << lane << " row=" << row;
        }
    }
}

TEST_F(MfmaTest, Mfma16x16x16_Identity)
{
    // Test with identity: A[m][k] = (m==k) ? 1 : 0, B[k][n] = (k==n) ? 1 : 0
    // So D = A * B = I * I = I.
    std::vector<fp16x4_t> a_host(WAVE_SIZE, fp16x4_t{0, 0, 0, 0});
    std::vector<fp16x4_t> b_host(WAVE_SIZE, fp16x4_t{0, 0, 0, 0});
    std::vector<fp32x4_t> c_ref(WAVE_SIZE);

    // A[row][k] = delta(row,k): a_host[(k/4)*16 + row][k%4] = 1 when row==k
    for(int row = 0; row < 16; row++)
    {
        int k        = row;
        int lane_idx = (k / 4) * 16 + row;
        a_host[lane_idx][k % 4] = static_cast<_Float16>(1.0f);
    }
    // B[k][col] = delta(k,col): b_host[(k/4)*16 + col][k%4] = 1 when k==col
    for(int col = 0; col < 16; col++)
    {
        int k        = col;
        int lane_idx = (k / 4) * 16 + col;
        b_host[lane_idx][k % 4] = static_cast<_Float16>(1.0f);
    }

    reference_mfma_16x16x16(a_host, b_host, c_ref);

    // D[m][n] = sum_k A[m][k] * B[k][n] = delta(m,n)
    // So c_ref should be identity in the output mapping.
    for(int lane = 0; lane < 64; lane++)
    {
        int lane_q  = lane % 16;
        int lane_c4 = lane / 16;
        for(int elem = 0; elem < 4; elem++)
        {
            int m        = lane_c4 * 4 + elem;
            float expected = (m == lane_q) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(c_ref[lane][elem], expected)
                << "Identity test: lane=" << lane << " elem=" << elem << " m=" << m
                << " n=" << lane_q;
        }
    }
}
