// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "compilation/Kernel.hpp"
#include "compilation/Program.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::compilation;

TEST(TestProgram, CompilesAndGetsKernel)
{
    SKIP_IF_NO_DEVICES();

    const Program program("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    hipFunction_t kernel = program.getKernel("vector_add");
    EXPECT_NE(nullptr, kernel);
}

TEST(TestProgram, InvalidProgramName)
{
    SKIP_IF_NO_DEVICES();
    EXPECT_THROW(Program("bad_filename.cpp", {"-O3", "-DFLOAT=float"}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestProgram, CompileFails)
{
    SKIP_IF_NO_DEVICES();

    // Missing "TYPE" macro definition
    EXPECT_THROW(Program("vector_add.cpp", {"-O3"}), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestKernel, LaunchesVectorAdd)
{
    SKIP_IF_NO_DEVICES();

    constexpr int N = 256;

    // Allocate and initialize
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&devA, N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&devB, N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&devC, N * sizeof(float)));

    std::vector<float> hostA(N, 1.0f);
    std::vector<float> hostB(N, 2.0f);
    std::vector<float> hostC(N);

    ASSERT_EQ(hipSuccess, hipMemcpy(devA, hostA.data(), N * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(devB, hostB.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    const Program program("vector_add.cpp", {"-O3", "-DFLOAT=float"});
    Kernel kernel(program, "vector_add");
    kernel.setBlockSize(256);
    kernel.setGridSize(1);
    kernel.launch(nullptr, devA, devB, devC, N);

    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_EQ(hipSuccess, hipMemcpy(hostC.data(), devC, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify
    EXPECT_FLOAT_EQ(3.0f, hostC[0]);
    EXPECT_FLOAT_EQ(3.0f, hostC[N - 1]);

    ASSERT_EQ(hipSuccess, hipFree(devA));
    ASSERT_EQ(hipSuccess, hipFree(devB));
    ASSERT_EQ(hipSuccess, hipFree(devC));
}

namespace
{

void freeDeviceBuffer(float* buffer)
{
    EXPECT_EQ(hipSuccess, hipFree(buffer));
}

void destroyStream(hipStream_t stream)
{
    EXPECT_EQ(hipSuccess, hipStreamDestroy(stream));
}

class TestKernelDeviceBinding : public ::testing::Test
{
protected:
    static constexpr int N = 256;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipSuccess, hipGetDeviceCount(&_deviceCount));
        ASSERT_EQ(hipSuccess, hipGetDevice(&_moduleDevice));

        for(auto* buffer : {&_a, &_b, &_c})
        {
            float* allocation = nullptr;
            const hipError_t status = hipMalloc(&allocation, N * sizeof(float));
            buffer->reset(allocation);
            ASSERT_EQ(hipSuccess, status);
        }
        const std::vector<float> hostA(N, 1.0f);
        const std::vector<float> hostB(N, 2.0f);
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(_a.get(), hostA.data(), N * sizeof(float), hipMemcpyHostToDevice));
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(_b.get(), hostB.data(), N * sizeof(float), hipMemcpyHostToDevice));
        ASSERT_EQ(hipSuccess, hipMemset(_c.get(), 0, N * sizeof(float)));
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
        _program = std::make_unique<Program>("vector_add.cpp",
                                             std::vector<std::string>{"-O3", "-DFLOAT=float"});
    }

    ~TestKernelDeviceBinding() override
    {
        // The fixture owns every HIP resource and restores the device captured in SetUp,
        // including when a fatal assertion or exception interrupts a cross-device test.
        if(_stream)
        {
            EXPECT_EQ(hipSuccess, hipSetDevice(_streamDevice));
            EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
            _stream.reset();
        }
        if(_moduleDevice >= 0)
        {
            EXPECT_EQ(hipSuccess, hipSetDevice(_moduleDevice));
            EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
            _program.reset();
            _a.reset();
            _b.reset();
            _c.reset();
        }
    }

    void createStream(int ordinal)
    {
        ASSERT_EQ(hipSuccess, hipSetDevice(ordinal));
        _streamDevice = ordinal;
        hipStream_t stream = nullptr;
        const hipError_t status = hipStreamCreate(&stream);
        _stream.reset(stream);
        ASSERT_EQ(hipSuccess, status);
    }

    void expectOutput(float expected)
    {
        ASSERT_EQ(hipSuccess, hipSetDevice(_moduleDevice));
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
        std::vector<float> output(N);
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(output.data(), _c.get(), N * sizeof(float), hipMemcpyDeviceToHost));
        for(std::size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_FLOAT_EQ(expected, output[i]) << "element " << i;
        }
    }

    int _deviceCount = 0;
    int _moduleDevice = -1;
    int _streamDevice = -1;
    std::unique_ptr<Program> _program;
    std::unique_ptr<float, decltype(&freeDeviceBuffer)> _a{nullptr, freeDeviceBuffer};
    std::unique_ptr<float, decltype(&freeDeviceBuffer)> _b{nullptr, freeDeviceBuffer};
    std::unique_ptr<float, decltype(&freeDeviceBuffer)> _c{nullptr, freeDeviceBuffer};
    std::unique_ptr<std::remove_pointer_t<hipStream_t>, decltype(&destroyStream)> _stream{
        nullptr, destroyStream};
};

enum class StreamKind
{
    NULL_STREAM,
    LEGACY,
    PER_THREAD,
    CONCRETE
};

class TestKernelStreamDeviceBinding
    : public TestKernelDeviceBinding,
      public ::testing::WithParamInterface<std::tuple<StreamKind, bool>>
{
};

TEST_P(TestKernelStreamDeviceBinding, LaunchesOnModuleDeviceAndRestoresCaller)
{
    const auto [kind, differentCurrentDevice] = GetParam();
    if(differentCurrentDevice && _deviceCount < 2)
    {
        GTEST_SKIP() << "cross-device launches require two devices";
    }

    hipStream_t stream = nullptr;
    switch(kind)
    {
    case StreamKind::NULL_STREAM:
        break;
    case StreamKind::LEGACY:
        stream = hipStreamLegacy;
        break;
    case StreamKind::PER_THREAD:
        stream = hipStreamPerThread;
        break;
    case StreamKind::CONCRETE:
        ASSERT_NO_FATAL_FAILURE(createStream(_moduleDevice));
        stream = _stream.get();
        break;
    default:
        FAIL() << "unknown stream kind";
    }

    Kernel kernel(_program->getKernel("vector_add"), "vector_add", _moduleDevice);
    kernel.setBlockSize(N);
    kernel.setGridSize(1);
    const int callerDevice
        = differentCurrentDevice ? (_moduleDevice + 1) % _deviceCount : _moduleDevice;
    ASSERT_EQ(hipSuccess, hipSetDevice(callerDevice));

    int afterLaunch = -1;
    kernel.launch(stream, _a.get(), _b.get(), _c.get(), N);
    ASSERT_EQ(hipSuccess, hipGetDevice(&afterLaunch));
    EXPECT_EQ(callerDevice, afterLaunch);
    expectOutput(3.0f);
}

INSTANTIATE_TEST_SUITE_P(StreamKinds,
                         TestKernelStreamDeviceBinding,
                         ::testing::Combine(::testing::Values(StreamKind::NULL_STREAM,
                                                              StreamKind::LEGACY,
                                                              StreamKind::PER_THREAD,
                                                              StreamKind::CONCRETE),
                                            ::testing::Bool()));

TEST_F(TestKernelDeviceBinding, RejectsAConcreteStreamFromAnotherDevice)
{
    if(_deviceCount < 2)
    {
        GTEST_SKIP() << "foreign-stream rejection requires two devices";
    }
    const int peerDevice = (_moduleDevice + 1) % _deviceCount;
    Kernel kernel(_program->getKernel("vector_add"), "vector_add", _moduleDevice);
    kernel.setBlockSize(N);
    kernel.setGridSize(1);
    ASSERT_NO_FATAL_FAILURE(createStream(peerDevice));

    int afterLaunch = -1;
    EXPECT_THROW(kernel.launch(_stream.get(), _a.get(), _b.get(), _c.get(), N),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    ASSERT_EQ(hipSuccess, hipGetDevice(&afterLaunch));
    EXPECT_EQ(peerDevice, afterLaunch);
    ASSERT_EQ(hipSuccess, hipSetDevice(peerDevice));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    expectOutput(0.0f);
}

TEST_F(TestKernelDeviceBinding, RefusesALaunchItCannotBindTheDeviceFor)
{
    // The loaded function and buffers are valid; only the requested ordinal is invalid.
    Kernel kernel(_program->getKernel("vector_add"), "vector_add", _deviceCount);
    kernel.setBlockSize(N);
    kernel.setGridSize(1);

    int afterLaunch = -1;
    EXPECT_THROW(kernel.launch(nullptr, _a.get(), _b.get(), _c.get(), N),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    const hipError_t deviceStatus = hipGetDevice(&afterLaunch);

    // A refused hipSetDevice leaves both HIP error slots populated.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
    ASSERT_EQ(hipSuccess, deviceStatus);
    EXPECT_EQ(_moduleDevice, afterLaunch);
    expectOutput(0.0f);
}

#ifdef HIP_KERNEL_PROVIDER_HAS_CLUSTER_LAUNCH
TEST_F(TestKernelDeviceBinding, RefusesAClusterLaunchOnADeviceWithoutClusters)
{
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipSuccess, hipGetDeviceProperties(&properties, _moduleDevice));
    if(properties.clusterLaunch != 0)
    {
        GTEST_SKIP() << "the device supports cluster launch";
    }
    Kernel kernel(_program->getKernel("vector_add"), "vector_add", _moduleDevice);
    kernel.setBlockSize(N);
    kernel.setGridSize(2);
    kernel.setClusterDims(2, 1, 1);

    // Refused rather than launched unclustered: a clustered kernel run that way reads
    // cluster ids that do not exist.
    try
    {
        kernel.launch(nullptr, _a.get(), _b.get(), _c.get(), N);
        FAIL() << "expected the cluster launch to be refused";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("does not support cluster launch"),
                  std::string::npos)
            << error.what();
    }
    expectOutput(0.0f);
}
#endif

// ---------------------------------------------------------------------------
// Cluster dimensions. Everything below throws before the first HIP call, so none of it
// needs a device.
// ---------------------------------------------------------------------------

std::string clusterRefusal(Kernel& kernel, unsigned int x, unsigned int y, unsigned int z)
{
    try
    {
        kernel.setClusterDims(x, y, z);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        return error.what();
    }
    return {};
}

TEST(TestKernelCluster, RefusesADimensionOutsideTheHardwareRange)
{
    Kernel kernel(nullptr, "clustered", Kernel::NO_DEVICE);

    for(const auto& dims : std::vector<std::array<unsigned int, 3>>{
            {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {16, 1, 1}, {1, 16, 1}, {1, 1, 16}})
    {
        const std::string message = clusterRefusal(kernel, dims[0], dims[1], dims[2]);
        EXPECT_NE(message.find("each dimension must be in 1..15"), std::string::npos)
            << "(" << dims[0] << ", " << dims[1] << ", " << dims[2] << "): " << message;
        EXPECT_NE(message.find("kernel 'clustered'"), std::string::npos) << message;
    }
}

TEST(TestKernelCluster, RefusesMoreThanSixteenWorkgroups)
{
    Kernel kernel(nullptr, "clustered", Kernel::NO_DEVICE);

    const std::string message = clusterRefusal(kernel, 4, 4, 2);
    EXPECT_NE(message.find("cluster_dims (4, 4, 2): 32 workgroups exceeds the cluster limit of 16"),
              std::string::npos)
        << message;
}

TEST(TestKernelCluster, AcceptsTheLimits)
{
    Kernel kernel(nullptr, "clustered", Kernel::NO_DEVICE);

    EXPECT_NO_THROW(kernel.setClusterDims(15, 1, 1));
    EXPECT_NO_THROW(kernel.setClusterDims(4, 4, 1));
    EXPECT_NO_THROW(kernel.setClusterDims(2, 2, 4));
    EXPECT_NO_THROW(kernel.setClusterDims(1, 1, 1));
}

TEST(TestKernelCluster, RefusesAGridThatIsNotAWholeNumberOfClusters)
{
    // A null function and no device: reaching HIP would fail differently, so the message
    // below also proves the check runs first.
    Kernel kernel(nullptr, "clustered", Kernel::NO_DEVICE);
    kernel.setGridSize(4, 3, 1);
    kernel.setClusterDims(2, 2, 1);

    try
    {
        kernel.launch(nullptr);
        FAIL() << "expected a partial cluster to be refused";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what())
                      .find("grid (4, 3, 1) is not a multiple of cluster (2, 2, 1) in y"),
                  std::string::npos)
            << error.what();
    }
}

TEST(TestKernelCluster, IsRefusedByAKernelWithoutClusterSupport)
{
    // The interface default: a kernel that cannot cluster must not launch unclustered.
    class PlainKernel : public IRunnableKernel
    {
    public:
        void setBlockSize(unsigned int, unsigned int, unsigned int) override {}
        void setGridSize(unsigned int, unsigned int, unsigned int) override {}
        void setSharedMemBytes(unsigned int) override {}

    protected:
        void launchImpl(hipStream_t, void**) const override {}
    };

    PlainKernel kernel;
    EXPECT_THROW(kernel.setClusterDims(2, 1, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
