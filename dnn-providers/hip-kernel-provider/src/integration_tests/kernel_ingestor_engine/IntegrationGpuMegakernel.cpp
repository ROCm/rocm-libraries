// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(ENABLE_MEGAKERNEL_ENGINE)

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

/**
 * @file IntegrationGpuMegakernel.cpp
 * @brief One megakernel code object served end to end: Graph.custom_op with the
 *        megakernel payload, plan build, execute on the handle stream.
 *
 * The object is the integration set's stand-in (test_descriptors/integration/megakernel):
 * same ABI and launch path as a FlyDSL megakernel, but it needs no model weights behind
 * `tab`. It writes one word per launched thread, so the buffer proves the launch ran with
 * the packed grid and block and received both addresses and nlayers. Whether a shipped
 * object computes the right logits is the exporter's check, not this one.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{
namespace
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

constexpr const char* STANDIN_ENGINE_NAME = "hipkernel:megakernel_standin";
constexpr const char* STANDIN_MODEL = "megakernel-standin-gfx942";
constexpr int64_t STANDIN_CUS = 304;
constexpr int64_t STANDIN_THREADS = 256;
constexpr int64_t NLAYERS = 64;
constexpr uint64_t TAB_MAGIC = 0x0123456789ABCDEFULL;
constexpr int64_t TAB_WORDS = 16;

#define ASSERT_HIP_OK(expr) ASSERT_EQ((expr), hipSuccess) << #expr

std::string payload(int64_t cus, const std::string& src)
{
    return R"({"schema":1,"model":")" + std::string(STANDIN_MODEL) + R"(","src":")" + src
           + R"(","arch":"gfx942","cus":)" + std::to_string(cus)
           + R"(,"kv":"ROCM_ATTN","block":784,"M":1,"tp":1,"rt":0,"lm_head":false,"nlayers":)"
           + std::to_string(NLAYERS) + "}";
}

class IntegrationGpuMegakernel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        hipDeviceProp_t properties{};
        ASSERT_HIP_OK(hipGetDeviceProperties(&properties, 0));
        if(std::string(properties.gcnArchName).rfind("gfx942", 0) != 0
           || properties.multiProcessorCount != STANDIN_CUS)
        {
            GTEST_SKIP() << "the stand-in object is packed for gfx942 with " << STANDIN_CUS
                         << " CUs; device is " << properties.gcnArchName << " with "
                         << properties.multiProcessorCount;
        }
        ASSERT_HIP_OK(hipStreamCreate(&_stream));
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    struct Built
    {
        std::shared_ptr<Graph> graph;
        std::shared_ptr<TensorAttributes> tab;
        std::shared_ptr<TensorAttributes> glob;
    };

    static Built makeGraph(const std::string& data, int64_t globWords)
    {
        Built built{std::make_shared<Graph>(), nullptr, nullptr};
        built.graph->set_name("megakernel_standin")
            .set_io_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);
        built.tab = std::make_shared<TensorAttributes>();
        built.tab->set_uid(1)
            .set_name("tab")
            .set_data_type(DataType::INT64)
            .set_dim({TAB_WORDS})
            .set_stride({1});
        built.glob = std::make_shared<TensorAttributes>();
        built.glob->set_uid(2)
            .set_name("glob")
            .set_data_type(DataType::INT64)
            .set_dim({globWords})
            .set_stride({1});
        CustomOpAttributes attributes;
        attributes.set_custom_op_id(std::string("megakernel.") + STANDIN_MODEL)
            .set_compute_data_type(DataType::FLOAT)
            .set_data(std::vector<uint8_t>(data.begin(), data.end()));
        built.graph->custom_op({built.tab, built.glob}, 0, attributes);
        built.graph->set_preferred_engine_id_ext(
            hipdnn_data_sdk::utilities::engineNameToId(STANDIN_ENGINE_NAME));
        return built;
    }

    hipStream_t _stream = nullptr;
    hipdnnHandle_t _handle = nullptr;
};

TEST_F(IntegrationGpuMegakernel, ExecutesThePackedObjectWithItsGridBlockAndArguments)
{
    const int64_t threads = STANDIN_CUS * STANDIN_THREADS;
    auto built = makeGraph(payload(STANDIN_CUS, "0123456789abcdef"), threads + 1);
    auto& graph = *built.graph;

    auto result = graph.validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph.build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph.create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph.check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph.build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    int64_t workspace = -1;
    ASSERT_EQ(graph.get_workspace_size(workspace).code, ErrorCode::OK);
    EXPECT_EQ(workspace, 0);

    std::vector<uint64_t> tabHost(TAB_WORDS, 0);
    tabHost[0] = TAB_MAGIC;
    void* tab = nullptr;
    void* glob = nullptr;
    ASSERT_HIP_OK(hipMalloc(&tab, tabHost.size() * sizeof(uint64_t)));
    ASSERT_HIP_OK(hipMalloc(&glob, static_cast<size_t>(threads + 1) * sizeof(uint64_t)));
    ASSERT_HIP_OK(
        hipMemcpy(tab, tabHost.data(), tabHost.size() * sizeof(uint64_t), hipMemcpyHostToDevice));
    ASSERT_HIP_OK(hipMemset(glob, 0, static_cast<size_t>(threads + 1) * sizeof(uint64_t)));

    std::unordered_map<int64_t, void*> variantPack{{built.tab->get_uid(), tab},
                                                   {built.glob->get_uid(), glob}};
    result = graph.execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_HIP_OK(hipStreamSynchronize(_stream));

    std::vector<uint64_t> globHost(static_cast<size_t>(threads + 1));
    ASSERT_HIP_OK(hipMemcpy(
        globHost.data(), glob, globHost.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
    EXPECT_EQ(hipFree(tab), hipSuccess);
    EXPECT_EQ(hipFree(glob), hipSuccess);

    int64_t wrong = 0;
    for(int64_t i = 0; i < threads; ++i)
    {
        const uint64_t want
            = (0x4D4BULL << 48) | (static_cast<uint64_t>(NLAYERS) << 32) | static_cast<uint64_t>(i);
        wrong += globHost[static_cast<size_t>(i)] != want ? 1 : 0;
    }
    EXPECT_EQ(wrong, 0) << "of " << threads << " threads (grid " << STANDIN_CUS << " x "
                        << STANDIN_THREADS << ")";
    EXPECT_EQ(globHost[static_cast<size_t>(threads)], TAB_MAGIC) << "tab address not received";
}

TEST_F(IntegrationGpuMegakernel, DeclinesAPayloadWhoseSrcNoObjectCarries)
{
    auto built = makeGraph(payload(STANDIN_CUS, "fedcba9876543210"), 8);
    auto& graph = *built.graph;
    ASSERT_EQ(graph.validate().code, ErrorCode::OK);
    ASSERT_EQ(graph.build_operation_graph(_handle).code, ErrorCode::OK);
    auto result = graph.create_execution_plans();
    if(result.code == ErrorCode::OK)
    {
        result = graph.check_support();
    }
    if(result.code == ErrorCode::OK)
    {
        result = graph.build_plans();
    }
    EXPECT_NE(result.code, ErrorCode::OK);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif
