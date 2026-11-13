// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "compilation/CompiledProgram.hpp"
#include "compilation/DeviceResolvedKernel.hpp"
#include "compilation/Program.hpp"

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace hip_kernel_provider::compilation
{
namespace
{

constexpr int N = 256;

const std::vector<std::string>& vectorAddOptions()
{
    static const std::vector<std::string> s_options{"-O3", "-DFLOAT=float"};
    return s_options;
}

std::string baseArchOf(int ordinal)
{
    hipDeviceProp_t properties{};
    if(hipGetDeviceProperties(&properties, ordinal) != hipSuccess)
    {
        return {};
    }
    return std::string(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
}

/// Every ordinal whose base architecture is device 0's, device 0 first.
std::vector<int> devicesLikeTheFirst()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess)
    {
        return {};
    }
    std::vector<int> ordinals;
    const std::string first = baseArchOf(0);
    for(int ordinal = 0; ordinal < count; ++ordinal)
    {
        if(baseArchOf(ordinal) == first)
        {
            ordinals.push_back(ordinal);
        }
    }
    return ordinals;
}

void freeDeviceBuffer(float* buffer)
{
    EXPECT_EQ(hipSuccess, hipFree(buffer));
}

void destroyStream(hipStream_t stream)
{
    EXPECT_EQ(hipSuccess, hipStreamDestroy(stream));
}

using DeviceBuffer = std::unique_ptr<float, decltype(&freeDeviceBuffer)>;
using Stream = std::unique_ptr<std::remove_pointer_t<hipStream_t>, decltype(&destroyStream)>;

/// Runs vector_add through @p kernel on a stream owned by @p ordinal, with device 0
/// current, and returns the output. The caller's current device is left at 0.
std::vector<float> runVectorAdd(const IRunnableKernel& kernel, int ordinal)
{
    std::vector<float> output(N, 0.0f);

    EXPECT_EQ(hipSuccess, hipSetDevice(ordinal));
    DeviceBuffer a{nullptr, freeDeviceBuffer};
    DeviceBuffer b{nullptr, freeDeviceBuffer};
    DeviceBuffer c{nullptr, freeDeviceBuffer};
    for(auto* buffer : {&a, &b, &c})
    {
        float* allocation = nullptr;
        EXPECT_EQ(hipSuccess, hipMalloc(&allocation, N * sizeof(float)));
        buffer->reset(allocation);
    }
    const std::vector<float> hostA(N, 1.0f);
    const std::vector<float> hostB(N, 2.0f);
    EXPECT_EQ(hipSuccess,
              hipMemcpy(a.get(), hostA.data(), N * sizeof(float), hipMemcpyHostToDevice));
    EXPECT_EQ(hipSuccess,
              hipMemcpy(b.get(), hostB.data(), N * sizeof(float), hipMemcpyHostToDevice));
    EXPECT_EQ(hipSuccess, hipMemset(c.get(), 0, N * sizeof(float)));
    hipStream_t rawStream = nullptr;
    EXPECT_EQ(hipSuccess, hipStreamCreate(&rawStream));
    const Stream stream{rawStream, destroyStream};

    // Launched with another device current: the kernel must pick the stream's device.
    EXPECT_EQ(hipSuccess, hipSetDevice(0));
    kernel.launch(stream.get(), a.get(), b.get(), c.get(), N);

    EXPECT_EQ(hipSuccess, hipStreamSynchronize(stream.get()));
    EXPECT_EQ(hipSuccess,
              hipMemcpy(output.data(), c.get(), N * sizeof(float), hipMemcpyDeviceToHost));

    // Freed on their own device.
    EXPECT_EQ(hipSuccess, hipSetDevice(ordinal));
    c.reset();
    b.reset();
    a.reset();
    EXPECT_EQ(hipSuccess, hipSetDevice(0));
    return output;
}

void expectVectorAddOutput(const std::vector<float>& output, int ordinal)
{
    for(std::size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_FLOAT_EQ(3.0f, output[i]) << "device " << ordinal << ", element " << i;
    }
}

class TestGpuDeviceResolvedKernel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipSuccess, hipGetDevice(&_callerDevice));
        ASSERT_EQ(hipSuccess, hipSetDevice(0));
        _binary = Program::compileBinary("vector_add.cpp", vectorAddOptions());
    }

    void TearDown() override
    {
        if(_callerDevice >= 0)
        {
            EXPECT_EQ(hipSuccess, hipSetDevice(_callerDevice));
        }
    }

    /// The program loaded on device 0, plus a loader that counts the loads it makes for
    /// any other device.
    CompiledProgram makeProgram()
    {
        auto first = std::make_shared<Program>("vector_add.cpp", _binary, 0);
        return {first, baseArchOf(0), [this](int ordinal) {
                    _loads.push_back(ordinal);
                    return std::make_shared<Program>("vector_add.cpp", _binary, ordinal);
                }};
    }

    std::vector<int> _loads;

private:
    int _callerDevice = -1;
    Program::Binary _binary;
};

TEST_F(TestGpuDeviceResolvedKernel, LaunchesOnEveryDeviceOfTheBuiltArchitecture)
{
    const CompiledProgram program = makeProgram();
    auto kernel = program.getKernel("vector_add");
    kernel->setBlockSize(N, 1, 1);
    kernel->setGridSize(1, 1, 1);

    const auto ordinals = devicesLikeTheFirst();
    for(const int ordinal : ordinals)
    {
        expectVectorAddOutput(runVectorAdd(*kernel, ordinal), ordinal);
        const auto& resolved = dynamic_cast<const DeviceResolvedKernel&>(*kernel);
        EXPECT_EQ(resolved.resolvedFor(ordinal).program->deviceOrdinal(), ordinal);
    }

    // Device 0 was loaded before the kernel existed; every other device loaded once.
    const std::vector<int> others(ordinals.begin() + 1, ordinals.end());
    EXPECT_EQ(_loads, others);
}

TEST_F(TestGpuDeviceResolvedKernel, ARepeatLaunchOnASeenDeviceLoadsNothing)
{
    const CompiledProgram program = makeProgram();
    auto kernel = program.getKernel("vector_add");
    kernel->setBlockSize(N, 1, 1);
    kernel->setGridSize(1, 1, 1);

    const int ordinal = devicesLikeTheFirst().back();
    expectVectorAddOutput(runVectorAdd(*kernel, ordinal), ordinal);
    const auto loadsAfterFirst = _loads.size();
    expectVectorAddOutput(runVectorAdd(*kernel, ordinal), ordinal);

    EXPECT_EQ(_loads.size(), loadsAfterFirst);
}

TEST_F(TestGpuDeviceResolvedKernel, RefusesADeviceOfAnotherArchitecture)
{
    const ProgramKernelResolver resolver(
        "gfx_not_this_device", "vector_add", [](int) -> std::shared_ptr<Program> {
            ADD_FAILURE() << "a device of another architecture must not be loaded";
            return nullptr;
        });

    try
    {
        resolver.forOrdinal(0);
        FAIL() << "expected a device of another architecture to be refused";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("vector_add"), std::string::npos) << error.what();
    }
}

TEST_F(TestGpuDeviceResolvedKernel, AProgramLoadsOnTheRequestedDevice)
{
    const int ordinal = devicesLikeTheFirst().back();
    const Program program(
        "vector_add.cpp", Program::compileBinary("vector_add.cpp", vectorAddOptions()), ordinal);

    EXPECT_EQ(program.deviceOrdinal(), ordinal);
    EXPECT_NE(program.getKernel("vector_add"), nullptr);

    // Neither the load nor the lookup leaves the caller on another device.
    int current = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&current));
    EXPECT_EQ(current, 0);
}

} // namespace
} // namespace hip_kernel_provider::compilation
