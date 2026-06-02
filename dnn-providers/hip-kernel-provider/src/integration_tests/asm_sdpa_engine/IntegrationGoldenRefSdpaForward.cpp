// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace
{

struct FwdGoldenRefTestCase
{
    std::string name;
    std::vector<int64_t> qDims;
    std::vector<int64_t> vDims;
    float tolerance;

    static std::string getName(const testing::TestParamInfo<FwdGoldenRefTestCase>& info)
    {
        return info.param.name;
    }
};

std::vector<uint8_t> loadBinFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file.is_open())
    {
        return {};
    }
    const auto size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

std::filesystem::path getGoldenDataDir()
{
    return getCurrentExecutableDirectory() / "../lib/golden_data/SdpaFwd/bf16_hd128_nomask_batch";
}

class GoldenRefSdpaFwdBf16 : public ::testing::TestWithParam<FwdGoldenRefTestCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        const auto pluginPath
            = std::filesystem::weakly_canonical(getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

} // namespace

TEST_P(GoldenRefSdpaFwdBf16, Correctness)
{
    const auto& testCase = GetParam();
    const auto baseDir = getGoldenDataDir();
    const auto basePath = baseDir / testCase.name;

    if(!std::filesystem::exists(basePath.string() + ".tensor0.bin"))
    {
        GTEST_SKIP() << "Golden data not found at: " << basePath;
    }

    const auto deviceString = hip_kernel_provider_common::getDeviceString(_stream);
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP() << "Skipped: ASM SDPA kernel requires gfx942 or gfx950, got " << deviceString;
    }

    const auto& qDims = testCase.qDims;
    const auto& vDims = testCase.vDims;
    const std::vector<int64_t> kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};

    // Build graph programmatically (same path as IntegrationGpuSdpaForward)
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim(qDims).set_stride(generateStrides(qDims)).set_data_type(DataType::BFLOAT16);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim(kDims).set_stride(generateStrides(kDims)).set_data_type(DataType::BFLOAT16);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim(vDims).set_stride(generateStrides(vDims)).set_data_type(DataType::BFLOAT16);

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("SdpaFwdGoldenRef");
    const float scale = 1.0f / std::sqrt(static_cast<float>(qDims.back()));
    sdpaAttrs.set_attn_scale_value(scale);

    auto [o, stats] = graph.sdpa(q, k, v, sdpaAttrs);
    o->set_output(true).set_data_type(DataType::BFLOAT16);

    const auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

    const auto buildResult = graph.build(_handle);
    ASSERT_EQ(buildResult.code, ErrorCode::OK) << buildResult.err_msg;

    // Get UIDs assigned by validate()
    const int64_t qUid = q->get_uid();
    const int64_t kUid = k->get_uid();
    const int64_t vUid = v->get_uid();
    const int64_t oUid = o->get_uid();

    // Create GPU tensor bundle from graph
    GraphTensorBundle gpuBundle;
    graph.visit([&](const INode& node) {
        for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
        {
            if(!tensorAttr->get_is_virtual()
               && gpuBundle.tensors.find(tensorAttr->get_uid()) == gpuBundle.tensors.end())
            {
                gpuBundle.tensors.insert(
                    {tensorAttr->get_uid(), createTensorFromAttribute(*tensorAttr)});
            }
        }
        for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
        {
            if(gpuBundle.tensors.find(tensorAttr->get_uid()) == gpuBundle.tensors.end())
            {
                gpuBundle.tensors.insert(
                    {tensorAttr->get_uid(), createTensorFromAttribute(*tensorAttr)});
            }
        }
    });

    // Load input tensors from golden .bin files
    // Golden UIDs: Q=0, K=1, V=2 -> Graph UIDs: qUid, kUid, vUid
    struct Mapping
    {
        int64_t goldenUid;
        int64_t graphUid;
    };
    for(const auto& [goldenUid, graphUid] : std::vector<Mapping>{{0, qUid}, {1, kUid}, {2, vUid}})
    {
        const auto binPath = basePath.string() + ".tensor" + std::to_string(goldenUid) + ".bin";
        const auto binData = loadBinFile(binPath);
        ASSERT_FALSE(binData.empty()) << "Failed to load: " << binPath;

        auto& tensor = *gpuBundle.tensors.at(graphUid);
        const auto expectedSize = tensor.elementCount() * sizeof(bfloat16);
        ASSERT_EQ(binData.size(), expectedSize) << "Size mismatch for golden uid " << goldenUid;

        std::memcpy(tensor.rawHostData(), binData.data(), binData.size());
        tensor.markHostModified();
    }

    // Load golden output (O = uid 3)
    const auto goldenPath = basePath.string() + ".tensor3.bin";
    const auto goldenData = loadBinFile(goldenPath);
    ASSERT_FALSE(goldenData.empty()) << "Failed to load golden output: " << goldenPath;

    // Execute on GPU
    int64_t workspaceSize = 0;
    const auto wsResult = graph.get_workspace_size(workspaceSize);
    ASSERT_EQ(wsResult.code, ErrorCode::OK) << wsResult.err_msg;
    const Workspace workspace(static_cast<size_t>(workspaceSize));

    auto variantPack = gpuBundle.toDeviceVariantPack();
    const auto execResult = graph.execute(_handle, variantPack, workspace.get());
    ASSERT_EQ(execResult.code, ErrorCode::OK) << execResult.err_msg;

    // Compare GPU output vs golden reference
    auto& gpuOutput = *gpuBundle.tensors.at(oUid);
    gpuOutput.markDeviceModified();

    const auto numElements = gpuOutput.elementCount();
    ASSERT_EQ(goldenData.size(), numElements * sizeof(bfloat16));

    const auto* goldenPtr = reinterpret_cast<const bfloat16*>(goldenData.data());
    const auto* gpuPtr = static_cast<const bfloat16*>(gpuOutput.rawHostData());

    size_t mismatchCount = 0;
    float maxAbsDiff = 0.0f;
    for(size_t i = 0; i < numElements; ++i)
    {
        const float diff
            = std::abs(static_cast<float>(goldenPtr[i]) - static_cast<float>(gpuPtr[i]));
        const float refAbs = std::abs(static_cast<float>(goldenPtr[i]));
        const float threshold = testCase.tolerance + testCase.tolerance * refAbs;
        if(diff > threshold)
        {
            ++mismatchCount;
        }
        maxAbsDiff = std::max(maxAbsDiff, diff);
    }

    EXPECT_EQ(mismatchCount, 0u) << "GPU output does not match golden reference for O tensor: "
                                 << mismatchCount << "/" << numElements
                                 << " elements exceed tolerance " << testCase.tolerance
                                 << ", max abs diff = " << maxAbsDiff;
}

namespace
{

auto getFwdGoldenRefTestCases() -> std::vector<FwdGoldenRefTestCase>
{
    return {
        {"Small", {2, 4, 256, 128}, {2, 4, 256, 128}, 1e-2f},
        {"Medium", {2, 4, 512, 128}, {2, 4, 512, 128}, 1e-2f},
        {"Gqa", {1, 8, 256, 128}, {1, 2, 256, 128}, 1e-2f},
    };
}

} // namespace

INSTANTIATE_TEST_SUITE_P(GoldenRef,
                         GoldenRefSdpaFwdBf16,
                         testing::ValuesIn(getFwdGoldenRefTestCases()),
                         FwdGoldenRefTestCase::getName);
