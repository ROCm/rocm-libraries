// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
namespace
{
using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
uint16_t bf16(float x)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &x, 4);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}
float fp32(uint16_t x)
{
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    float value = 0;
    std::memcpy(&value, &bits, 4);
    return value;
}
class IntegrationGpuAttentionDecode : public ::testing::TestWithParam<bool>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    std::vector<void*> _allocations;
    void SetUp() override
    {
        const auto expectedArch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        if(expectedArch.empty())
        {
            SKIP_IF_NO_DEVICES();
        }
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        const auto pluginPath
            = std::filesystem::weakly_canonical(
                  hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH)
                  .string();
        const char* path = pluginPath.c_str();
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(1, &path, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
        int device = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        hipDeviceProp_t props{};
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        const std::string arch = props.gcnArchName;
        if(!expectedArch.empty())
        {
            ASSERT_EQ(arch.substr(0, arch.find(':')), expectedArch);
        }
        if(arch.find("gfx942") != 0 && arch.find("gfx950") != 0)
        {
            GTEST_SKIP() << "AttentionDecode supports gfx942/gfx950; found " << arch;
        }
        std::cout << "DECODE_DEVICE " << arch << '\n';
    }
    void TearDown() override
    {
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        }
        for(void* p : _allocations)
        {
            EXPECT_EQ(hipFree(p), hipSuccess);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }
    void* allocate(size_t bytes)
    {
        void* p = nullptr;
        if(hipMalloc(&p, bytes) != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed");
        }
        _allocations.push_back(p);
        return p;
    }
};
TEST_P(IntegrationGpuAttentionDecode, InitializesAndReusesCacheAcrossPageBoundary)
{
    Graph graph;
    graph.set_io_data_type(DataType::BFLOAT16)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
    auto tensor = [](int64_t uid, int64_t heads, int64_t seq) {
        return std::make_shared<TensorAttributes>(
            TensorAttributes()
                .set_uid(uid)
                .set_data_type(DataType::BFLOAT16)
                .set_dim({1, heads, seq, 128})
                .set_stride({seq * heads * 128, 128, heads * 128, 1}));
    };
    auto q = tensor(1, 8, 1);
    auto k = tensor(2, 2, 128);
    auto v = tensor(3, 2, 128);
    auto length = std::make_shared<TensorAttributes>(TensorAttributes()
                                                         .set_uid(5)
                                                         .set_data_type(DataType::INT32)
                                                         .set_dim({1, 1, 1, 1})
                                                         .set_stride({1, 1, 1, 1}));
    SdpaAttributes attrs;
    attrs.set_attn_scale(1.0F / std::sqrt(128.0F)).set_seq_len_kv(length).set_generate_stats(false);
    if(GetParam())
    {
        attrs.set_diagonal_band_right_bound(0).set_diagonal_alignment(
            DiagonalAlignment::BOTTOM_RIGHT);
    }
    auto o = graph.sdpa(q, k, v, attrs)[0];
    o->set_uid(4)
        .set_output(true)
        .set_data_type(DataType::BFLOAT16)
        .set_dim({1, 8, 1, 128})
        .set_stride({1024, 128, 1024, 1});
    const auto ok = [](const Error& error) {
        EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
        return error.code == ErrorCode::OK;
    };
    ASSERT_TRUE(ok(graph.validate()));
    ASSERT_TRUE(ok(graph.build_operation_graph(_handle)));
    const auto engine = hipdnn_data_sdk::utilities::engineNameToId("hipkernel:AttentionDecode");
    ASSERT_TRUE(ok(graph.create_execution_plan_ext(engine, {}))); // No fallback engine.
    ASSERT_TRUE(ok(graph.check_support()));
    ASSERT_TRUE(ok(graph.build_plans()));
    int64_t workspaceBytes = 0;
    ASSERT_TRUE(ok(graph.get_workspace_size(workspaceBytes)));
    ASSERT_EQ(workspaceBytes, 24); // [0,1,pad,pad] plus two identity-page entries.
    void* workspace = allocate(static_cast<size_t>(workspaceBytes));
    std::unordered_map<int64_t, void*> buffers{{1, allocate(2048)},
                                               {2, allocate(65536)},
                                               {3, allocate(65536)},
                                               {4, allocate(2048)},
                                               {5, allocate(4)}};
    std::vector<uint16_t> hq(1024);
    std::vector<uint16_t> hk(32768);
    std::vector<uint16_t> hv(32768);
    std::vector<uint16_t> ho(1024);
    float maxError = 0;
    // Reuse the same plan, buffers and workspace for a second sequence. Poison
    // before each reset so the initialization contract cannot pass by accident.
    for(int sequence = 0; sequence < 2; ++sequence)
    {
        SCOPED_TRACE(sequence);
        std::fill(hk.begin(), hk.end(), uint16_t{0x7fc0});
        std::fill(hv.begin(), hv.end(), uint16_t{0x7fc0});
        ASSERT_EQ(hipMemcpyAsync(buffers[2], hk.data(), 65536, hipMemcpyHostToDevice, _stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(buffers[3], hv.data(), 65536, hipMemcpyHostToDevice, _stream),
                  hipSuccess);
        // Required once per new sequence, ordered before appends and execution.
        ASSERT_EQ(hipMemsetAsync(buffers[2], 0, 65536, _stream), hipSuccess);
        ASSERT_EQ(hipMemsetAsync(buffers[3], 0, 65536, _stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        std::fill(hk.begin(), hk.end(), uint16_t{0});
        std::fill(hv.begin(), hv.end(), uint16_t{0});
        for(int32_t step = 1; step <= 128; ++step)
        {
            SCOPED_TRACE(step);
            const size_t base = static_cast<size_t>(step - 1) * 256;
            for(size_t i = 0; i < 256; ++i)
            {
                hk[base + i] = bf16(
                    std::sin(static_cast<float>(base + i) * 0.037F + static_cast<float>(sequence)));
                hv[base + i] = bf16(
                    std::cos(static_cast<float>(base + i) * 0.023F + static_cast<float>(sequence)));
            }
            for(size_t i = 0; i < hq.size(); ++i)
            {
                hq[i] = bf16(
                    std::sin(static_cast<float>(i) * 0.019F + static_cast<float>(step) * 0.13F));
            }
            ASSERT_EQ(hipMemcpyAsync(static_cast<uint16_t*>(buffers[2]) + base,
                                     hk.data() + base,
                                     512,
                                     hipMemcpyHostToDevice,
                                     _stream),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(static_cast<uint16_t*>(buffers[3]) + base,
                                     hv.data() + base,
                                     512,
                                     hipMemcpyHostToDevice,
                                     _stream),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(buffers[1], hq.data(), 2048, hipMemcpyHostToDevice, _stream),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(buffers[5], &step, 4, hipMemcpyHostToDevice, _stream),
                      hipSuccess);
            ASSERT_TRUE(ok(graph.execute(_handle, buffers, workspace)));
            ASSERT_EQ(hipMemcpyAsync(ho.data(), buffers[4], 2048, hipMemcpyDeviceToHost, _stream),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
            for(size_t h = 0; h < 8; ++h)
            {
                std::vector<float> scores(static_cast<size_t>(step));
                for(size_t t = 0; t < scores.size(); ++t)
                {
                    float dot = 0;
                    for(size_t d = 0; d < 128; ++d)
                    {
                        dot += fp32(hq[h * 128 + d]) * fp32(hk[t * 256 + (h / 4) * 128 + d]);
                    }
                    scores[t] = dot / std::sqrt(128.0F);
                }
                const float maximum = *std::max_element(scores.begin(), scores.end());
                float sum = 0;
                for(float& score : scores)
                {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                for(size_t d = 0; d < 128; ++d)
                {
                    float expected = 0;
                    for(size_t t = 0; t < scores.size(); ++t)
                    {
                        expected += scores[t] * fp32(hv[t * 256 + (h / 4) * 128 + d]) / sum;
                    }
                    const float actual = fp32(ho[h * 128 + d]);
                    ASSERT_TRUE(std::isfinite(actual));
                    const float error = std::abs(expected - actual);
                    maxError = std::max(maxError, error);
                    ASSERT_LE(error, 0.015F + 0.015F * std::abs(expected))
                        << "head=" << h << " d=" << d;
                }
            }
        }
    }
    std::cout << "DECODE_RESULT engine=hipkernel:AttentionDecode sequences=2 steps=256 "
                 "elements=262144 max_abs_error="
              << maxError << '\n';
}
INSTANTIATE_TEST_SUITE_P(Masks, IntegrationGpuAttentionDecode, ::testing::Values(false, true));
} // namespace
#endif
