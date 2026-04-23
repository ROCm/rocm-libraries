// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include "conv2d_gtest.hpp"

#if MIOPEN_EMBED_DB
#include "get_handle.hpp"
#endif // MIOPEN_EMBED_DB

namespace {

using TestCase = Conv2DBaseTestCase<NamedContainer<std::vector<size_t>>, // input_dims
                                    NamedContainer<std::vector<size_t>>  // weights_tensor_dims
                                    >;

template <typename T>
auto GenCases(bool smoke_test,
              std::vector<size_t> input_dims,
              std::vector<size_t> weight_tensor_dims,
              std::vector<int> pads_strides_dilations)
{
    Conv2DBaseTestParameters<T> baseParams(smoke_test);

    baseParams.pads_strides_dilations = {std::move(pads_strides_dilations)};

    return conv2d_test_base<T>::GenTestParams(
        baseParams,
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "input_dims", std::vector<std::vector<size_t>>{std::move(input_dims)}),
        MakeNamedParameterCollectionValues<std::vector<size_t>>(
            "weight_tensor_dims", std::vector<std::vector<size_t>>{std::move(weight_tensor_dims)}));
}

#if MIOPEN_EMBED_DB
bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    const std::string devName = handle.GetDeviceName();
    return (devName == "gfx900" || devName == "gfx906");
}
#endif // MIOPEN_EMBED_DB

} // namespace

template <typename T>
struct conv_embed_db_test : public conv2d_test_base<T, TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        this->GetTestParams(this->input_dims, this->weight_tensor_dims);
    }
};

using CPU_ConvEmbedConfig_BFP16 = conv_embed_db_test<bfloat16>;
using CPU_ConvEmbedConfig_FP16  = conv_embed_db_test<half_float::half>;
using CPU_ConvEmbedConfig_FP32  = conv_embed_db_test<float>;
using CPU_ConvEmbedConfig_INT8  = conv_embed_db_test<int8_t>;

TEST_P(CPU_ConvEmbedConfig_BFP16, TestBFloat16)
{
#if MIOPEN_EMBED_DB
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        testing::internal::CaptureStderr();
        run();
        const auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
#else  // MIOPEN_EMBED_DB
    GTEST_SKIP() << "Test disabled at compile time";
#endif // MIOPEN_EMBED_DB
}

TEST_P(CPU_ConvEmbedConfig_FP16, TestFloat16)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        testing::internal::CaptureStderr();
        run();
        const auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }

#else  // MIOPEN_EMBED_DB
    GTEST_SKIP() << "Test disabled at compile time";
#endif // MIOPEN_EMBED_DB
}

TEST_P(CPU_ConvEmbedConfig_FP32, TestFloat32)
{
#if MIOPEN_EMBED_DB
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        testing::internal::CaptureStderr();
        run();
        const auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
#else  // MIOPEN_EMBED_DB
    GTEST_SKIP() << "Test disabled at compile time";
#endif // MIOPEN_EMBED_DB
}

TEST_P(CPU_ConvEmbedConfig_INT8, TestInt8)
{
#if MIOPEN_EMBED_DB
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        testing::internal::CaptureStderr();
        run();
        const auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
    else
    {
        GTEST_SKIP() << "Test not supported for the current device";
    }
#else  // MIOPEN_EMBED_DB
    GTEST_SKIP() << "Test disabled at compile time";
#endif // MIOPEN_EMBED_DB
}

#define INSTANTIATE_ALL_TEST_SUITES(id, ...)                                              \
    INSTANTIATE_TEST_SUITES(id, CPU_ConvEmbedConfig_BFP16, bfloat16, __VA_ARGS__);        \
    INSTANTIATE_TEST_SUITES(id, CPU_ConvEmbedConfig_FP16, half_float::half, __VA_ARGS__); \
    INSTANTIATE_TEST_SUITES(id, CPU_ConvEmbedConfig_FP32, float, __VA_ARGS__);            \
    INSTANTIATE_TEST_SUITES(id, CPU_ConvEmbedConfig_INT8, int8_t, __VA_ARGS__)

INSTANTIATE_ALL_TEST_SUITES(0, {128, 128, 28, 28}, {128, 128, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(1, {128, 256, 56, 56}, {512, 256, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(2, {128, 3, 230, 230}, {64, 3, 7, 7}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(3, {128, 64, 56, 56}, {64, 64, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(4, {128, 256, 14, 14}, {256, 256, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(5, {128, 512, 7, 7}, {512, 512, 3, 3}, {1, 1, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(6, {128, 1024, 14, 14}, {512, 1024, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(7, {128, 1024, 14, 14}, {2048, 1024, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(8, {128, 256, 14, 14}, {1024, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(9, {128, 512, 28, 28}, {256, 512, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(10, {128, 1024, 14, 14}, {256, 1024, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(11, {128, 64, 56, 56}, {256, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(12, {128, 64, 56, 56}, {64, 64, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(13, {128, 128, 28, 28}, {512, 128, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(14, {128, 256, 56, 56}, {128, 256, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(15, {128, 256, 56, 56}, {64, 256, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(16, {128, 512, 28, 28}, {1024, 512, 1, 1}, {0, 0, 2, 2, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(17, {128, 512, 28, 28}, {128, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(18, {128, 512, 7, 7}, {2048, 512, 1, 1}, {0, 0, 1, 1, 1, 1});
INSTANTIATE_ALL_TEST_SUITES(19, {128, 2048, 7, 7}, {512, 2048, 1, 1}, {0, 0, 1, 1, 1, 1});
