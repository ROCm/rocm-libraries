// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the comparison helpers declared in
// clients/common/include/unit.hpp. These do not require a GPU.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "unit.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
    float float_from_bits(uint32_t bits)
    {
        static_assert(sizeof(float) == sizeof(bits));
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    TEST(UnitCheckIdentical, compares_dense_strided_batches)
    {
        constexpr int64_t  M = 2, N = 3, stride = M * N, batches = 2;
        std::vector<float> cpu(stride * batches);
        for(size_t i = 0; i < cpu.size(); ++i)
            cpu[i] = static_cast<float>(i);
        std::vector<float> gpu = cpu;

        EXPECT_TRUE(unit_check_storage_identical(M, N, M, stride, cpu.data(), gpu.data(), batches));
        gpu.back() += 1.0f;
        EXPECT_FALSE(
            unit_check_storage_identical(M, N, M, stride, cpu.data(), gpu.data(), batches));
    }

    TEST(UnitCheckIdentical, compares_zero_stride_once_for_multiple_batches)
    {
        constexpr int64_t M = 2, N = 2, batches = 3;
        // A zero stride aliases every logical batch to this one matrix.
        std::vector<float> cpu{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> gpu = cpu;

        EXPECT_TRUE(unit_check_storage_identical(M, N, M, 0, cpu.data(), gpu.data(), batches));
        gpu.back() += 1.0f;
        EXPECT_FALSE(unit_check_storage_identical(M, N, M, 0, cpu.data(), gpu.data(), batches));
    }

    TEST(UnitCheckIdentical, rejects_mismatched_element_types)
    {
        const float cpu = 1.0f;
        static_assert(sizeof(float) % sizeof(hip_bfloat16) == 0);
        std::array<hip_bfloat16, sizeof(float) / sizeof(hip_bfloat16)> gpu;
        // Keep the storage equal so only the element-type check can reject it.
        std::memcpy(gpu.data(), &cpu, sizeof(cpu));

        EXPECT_FALSE(unit_check_storage_identical(1, 1, 1, 0, &cpu, gpu.data(), 1));
    }

    TEST(UnitCheckIdentical, ignores_column_padding_and_batch_gaps)
    {
        constexpr int64_t  M = 2, N = 2, lda = 3, stride = 8, batches = 2;
        std::vector<float> cpu(stride * batches, 1.0f);
        std::vector<float> gpu = cpu;
        for(int64_t batch = 0; batch < batches; ++batch)
        {
            gpu[batch * stride + 2] = 2.0f;
            gpu[batch * stride + 5] = 3.0f;
            gpu[batch * stride + 6] = 4.0f;
            gpu[batch * stride + 7] = 5.0f;
        }

        EXPECT_TRUE(
            unit_check_storage_identical(M, N, lda, stride, cpu.data(), gpu.data(), batches));
        gpu[stride + lda] = 6.0f;
        EXPECT_FALSE(
            unit_check_storage_identical(M, N, lda, stride, cpu.data(), gpu.data(), batches));
    }

    TEST(UnitCheckIdentical, falls_back_for_numerically_equal_encodings)
    {
        const std::vector<float> cpu{
            0.0f,
            float_from_bits(0x7fc00001),
        };
        const std::vector<float> gpu{
            -0.0f,
            float_from_bits(0x7fc00002),
        };

        EXPECT_FALSE(unit_check_storage_identical(1, 2, 1, 0, cpu.data(), gpu.data(), 1));
        EXPECT_NO_FATAL_FAILURE(unit_check_general<float>(1, 2, 1, 0, cpu.data(), gpu.data(), 1));
    }

    TEST(UnitCheckIdentical, supports_pointer_array_batches)
    {
        const std::vector<float> cpu0{1.0f, 2.0f};
        const std::vector<float> cpu1{3.0f, 4.0f};
        std::vector<float>       gpu0  = cpu0;
        std::vector<float>       gpu1  = cpu1;
        const float*             cpu[] = {cpu0.data(), cpu1.data()};
        const float*             gpu[] = {gpu0.data(), gpu1.data()};

        EXPECT_TRUE(unit_check_batched_storage_identical(2, 1, 2, cpu, gpu, 2));
        gpu1[1] += 1.0f;
        EXPECT_FALSE(unit_check_batched_storage_identical(2, 1, 2, cpu, gpu, 2));
    }

    void run_unit_check_mismatch()
    {
        const float cpu = 1.0f;
        const float gpu = 2.0f;
        unit_check_general<float>(1, 1, 1, 0, &cpu, &gpu, 1);
    }

    TEST(UnitCheckIdentical, public_unit_check_rejects_mismatch)
    {
        EXPECT_FATAL_FAILURE(run_unit_check_mismatch(), "Expected equality");
    }

    void run_batched_unit_check_mismatch()
    {
        const float  cpu_value   = 1.0f;
        const float  gpu_value   = 2.0f;
        const float* cpu_batch[] = {&cpu_value};
        const float* gpu_batch[] = {&gpu_value};
        unit_check_general<float>(1, 1, 1, cpu_batch, gpu_batch, 1);
    }

    TEST(UnitCheckIdentical, public_batched_unit_check_rejects_mismatch)
    {
        EXPECT_FATAL_FAILURE(run_batched_unit_check_mismatch(), "Expected equality");
    }

    void run_special_value_mismatch()
    {
        float cpu = 1.0f;
        float gpu = std::numeric_limits<float>::infinity();
        check_special_value_consistency(1, 1, 1, 0, &cpu, &gpu, 1, HIP_R_32F);
    }

    TEST(UnitCheckIdentical, special_value_check_rejects_mismatch)
    {
        EXPECT_FATAL_FAILURE(run_special_value_mismatch(), "Special value mismatch");
    }

} // namespace
