// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkValidation.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

/**
 * @file TestBenchmarkValidationHip.cpp
 * @brief Device tests for hipValidationHooks(): the real fill and snapshot on device buffers,
 *        and through a BenchmarkPlan whose injected timer leaves no post-warmup drain, so the
 *        snapshot hook itself must synchronize the stream.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
using DeviceBuffer = hipdnn_data_sdk::utilities::ScopedResource<void*>;

DeviceBuffer allocateDevice(size_t bytes)
{
    void* raw = nullptr;
    if(hipMalloc(&raw, bytes) != hipSuccess || raw == nullptr)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    return {raw, [](void* ptr) { static_cast<void>(hipFree(ptr)); }};
}

hipdnn_data_sdk::utilities::ScopedResource<hipStream_t> createStream()
{
    hipStream_t stream = nullptr;
    if(hipStreamCreate(&stream) != hipSuccess)
    {
        throw std::runtime_error("hipStreamCreate failed");
    }
    return {stream, [](hipStream_t handle) { static_cast<void>(hipStreamDestroy(handle)); }};
}

TEST(TestIngestorBenchmarkValidationHip, FillThenSnapshotRoundTripsEveryDtypesFillByte)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t FLOAT_BYTES = 64;
    constexpr size_t FNUZ_BYTES = 16;
    const auto floatBuffer = allocateDevice(FLOAT_BYTES);
    const auto fnuzBuffer = allocateDevice(FNUZ_BYTES);
    const std::vector<hipdnnPluginDeviceBuffer_t> buffers{{1, floatBuffer.get()},
                                                          {2, fnuzBuffer.get()}};
    const std::vector<OutputBufferSpec> specs{{1, DataType::FLOAT, FLOAT_BYTES},
                                              {2, DataType::FP8_E4M3_FNUZ, FNUZ_BYTES}};
    const auto stream = createStream();
    const auto hooks = hipValidationHooks();

    // Poison the buffers so the fill is what the snapshot sees.
    ASSERT_EQ(hipMemset(floatBuffer.get(), 0x11, FLOAT_BYTES), hipSuccess);
    ASSERT_EQ(hipMemset(fnuzBuffer.get(), 0x11, FNUZ_BYTES), hipSuccess);

    hooks.fill(stream.get(), buffers.data(), static_cast<uint32_t>(buffers.size()), specs);
    OutputImages images;
    ASSERT_TRUE(hooks.snapshot(
        stream.get(), buffers.data(), static_cast<uint32_t>(buffers.size()), specs, images));

    ASSERT_EQ(images.size(), 2U);
    EXPECT_EQ(images[0], std::vector<uint8_t>(FLOAT_BYTES, 0xFF));
    EXPECT_EQ(images[1], std::vector<uint8_t>(FNUZ_BYTES, 0x80));
    EXPECT_TRUE(hipdnn_plugin_sdk::output_agreement::imageUntouched(DataType::FLOAT, images[0]));
    EXPECT_TRUE(
        hipdnn_plugin_sdk::output_agreement::imageUntouched(DataType::FP8_E4M3_FNUZ, images[1]));
}

TEST(TestIngestorBenchmarkValidationHip, ASnapshotReportsAMissingBufferAsAFailedCopy)
{
    SKIP_IF_NO_DEVICES();

    const auto stream = createStream();
    const auto hooks = hipValidationHooks();
    const std::vector<OutputBufferSpec> specs{{7, DataType::FLOAT, 16}};
    OutputImages images;
    EXPECT_FALSE(hooks.snapshot(stream.get(), nullptr, 0, specs, images));
}

// The second copy is rejected by the runtime because it reads past its allocation, after the
// first copy was already enqueued. The hook must still drain the stream and report failure,
// and the first image is whole by the time it returns.
TEST(TestIngestorBenchmarkValidationHip, ASnapshotReportsAFailedSecondCopyAndKeepsTheFirstImage)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t FIRST_BYTES = 64;
    constexpr size_t SECOND_BYTES = 16;
    constexpr size_t OVERSIZED_BYTES = 1U << 16;
    const auto first = allocateDevice(FIRST_BYTES);
    const auto second = allocateDevice(SECOND_BYTES);
    ASSERT_EQ(hipMemset(first.get(), 0x5A, FIRST_BYTES), hipSuccess);
    ASSERT_EQ(hipMemset(second.get(), 0x22, SECOND_BYTES), hipSuccess);
    const std::vector<hipdnnPluginDeviceBuffer_t> buffers{{1, first.get()}, {2, second.get()}};
    const std::vector<OutputBufferSpec> specs{{1, DataType::FLOAT, FIRST_BYTES},
                                              {2, DataType::FLOAT, OVERSIZED_BYTES}};
    const auto stream = createStream();
    const auto hooks = hipValidationHooks();

    OutputImages images;
    EXPECT_FALSE(hooks.snapshot(
        stream.get(), buffers.data(), static_cast<uint32_t>(buffers.size()), specs, images));

    ASSERT_EQ(images.size(), 2U);
    EXPECT_EQ(images[0], std::vector<uint8_t>(FIRST_BYTES, 0x5A));
}

TEST(TestIngestorBenchmarkValidationHip,
     ASnapshotOfAZeroByteOutputYieldsAnEmptyImageAndCopiesTheRest)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t BYTES = 32;
    const auto empty = allocateDevice(BYTES);
    const auto filled = allocateDevice(BYTES);
    ASSERT_EQ(hipMemset(filled.get(), 0x3C, BYTES), hipSuccess);
    const std::vector<hipdnnPluginDeviceBuffer_t> buffers{{1, empty.get()}, {2, filled.get()}};
    const std::vector<OutputBufferSpec> specs{{1, DataType::FLOAT, 0}, {2, DataType::FLOAT, BYTES}};
    const auto stream = createStream();
    const auto hooks = hipValidationHooks();

    OutputImages images;
    ASSERT_TRUE(hooks.snapshot(
        stream.get(), buffers.data(), static_cast<uint32_t>(buffers.size()), specs, images));

    ASSERT_EQ(images.size(), 2U);
    EXPECT_TRUE(images[0].empty());
    EXPECT_EQ(images[1], std::vector<uint8_t>(BYTES, 0x3C));
}

/// A handle with a real stream, which the hooks and the plan's timer path use.
struct StreamHandle
{
    hipStream_t stream = nullptr;

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    hipStream_t getStream() const
    {
        return stream;
    }
};

/// Writes its byte to the bound buffer with an undrained, stream-ordered memset; a plan
/// without a byte never writes.
class MemsetPlan : public hipdnn_plugin_sdk::IPlan<StreamHandle>
{
public:
    MemsetPlan(std::optional<uint8_t> byte, size_t bytes)
        : _byte(byte)
        , _bytes(bytes)
    {
    }

    size_t getWorkspaceSize(const StreamHandle& /*handle*/) const override
    {
        return 0;
    }

    void execute(const StreamHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* /*workspace*/ = nullptr) const override
    {
        if(!_byte.has_value())
        {
            return;
        }
        void* ptr = hipdnn_plugin_sdk::ingestor::detail::findDeviceBuffer(
            deviceBuffers, numDeviceBuffers, 1);
        if(ptr == nullptr || hipMemsetAsync(ptr, *_byte, _bytes, handle.getStream()) != hipSuccess)
        {
            throw std::runtime_error("MemsetPlan: enqueue failed");
        }
    }

private:
    std::optional<uint8_t> _byte;
    size_t _bytes;
};

TEST(TestIngestorBenchmarkValidationHip,
     ANonWritingCandidateIsExcludedThroughTheRealHooksAndAnInjectedTimer)
{
    SKIP_IF_NO_DEVICES();

    using Plan = BenchmarkPlan<StreamHandle>;
    constexpr size_t OUTPUT_BYTES = 64;
    const auto output = allocateDevice(OUTPUT_BYTES);
    const std::vector<hipdnnPluginDeviceBuffer_t> buffers{{1, output.get()}};
    const auto stream = createStream();
    const StreamHandle handle{stream.get()};

    auto nonWriter = std::make_unique<MemsetPlan>(std::nullopt, OUTPUT_BYTES);
    auto writer = std::make_unique<MemsetPlan>(uint8_t{0x3C}, OUTPUT_BYTES);
    const hipdnn_plugin_sdk::IPlan<StreamHandle>* nonWriterRaw = nonWriter.get();
    const hipdnn_plugin_sdk::IPlan<StreamHandle>* writerRaw = writer.get();

    DescriptorId nonWriterId{};
    nonWriterId.fill(0x10);
    DescriptorId writerId{};
    writerId.fill(0x11);
    std::vector<Plan::Candidate> candidates;
    candidates.push_back({nonWriterId, std::move(nonWriter)});
    candidates.push_back({writerId, std::move(writer)});

    // The nonwriter is the fastest, so only validation can keep it from winning. The injected
    // timer never drains the stream: the snapshot hook has to.
    const std::map<const hipdnn_plugin_sdk::IPlan<StreamHandle>*, double> durations{
        {nonWriterRaw, 1.0}, {writerRaw, 2.0}};
    Plan::Timer timer = [durations](const hipdnn_plugin_sdk::IPlan<StreamHandle>& plan,
                                    const StreamHandle& planHandle,
                                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                    uint32_t numDeviceBuffers,
                                    void* workspace) -> std::optional<double> {
        plan.execute(planHandle, deviceBuffers, numDeviceBuffers, workspace);
        return durations.at(&plan);
    };

    std::vector<WinnerRecord> records;
    BenchmarkValidation validation;
    validation.outputs = {{1, DataType::FLOAT, OUTPUT_BYTES}};
    validation.hooks = hipValidationHooks();
    validation.record = [&records](WinnerRecord record) { records.push_back(std::move(record)); };

    const Plan plan(std::move(candidates),
                    handle,
                    std::move(timer),
                    {},
                    "hip-validation-test",
                    "device-0",
                    std::move(validation));
    plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);

    EXPECT_EQ(plan.chosenPlan(), writerRaw);
    ASSERT_EQ(records.size(), 1U);
    ASSERT_EQ(records.front().entries.size(), 1U);
    EXPECT_EQ(records.front().entries.front().kernelId, writerId);
    EXPECT_TRUE(records.front().validated.has_value());

    // The delegated execute is stream-ordered too: drain, then read the winner's output back.
    ASSERT_EQ(hipStreamSynchronize(stream.get()), hipSuccess);
    std::vector<uint8_t> result(OUTPUT_BYTES);
    ASSERT_EQ(hipMemcpy(result.data(), output.get(), OUTPUT_BYTES, hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(result, std::vector<uint8_t>(OUTPUT_BYTES, 0x3C));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
