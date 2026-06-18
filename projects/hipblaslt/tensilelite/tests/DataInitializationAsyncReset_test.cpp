// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>

#include <hip/hip_runtime.h>

#include <Tensile/Utils.hpp>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::makePlainProblem;

    struct StreamHandle
    {
        hipStream_t stream = nullptr;

        explicit StreamHandle(unsigned int flags = hipStreamNonBlocking)
        {
            HIP_CHECK_EXC(hipStreamCreateWithFlags(&stream, flags));
        }

        ~StreamHandle()
        {
            if(stream)
                (void)hipStreamDestroy(stream);
        }

        StreamHandle(StreamHandle const&)            = delete;
        StreamHandle& operator=(StreamHandle const&) = delete;

        hipStream_t get() const
        {
            return stream;
        }
    };

    struct EventHandle
    {
        hipEvent_t event = nullptr;

        explicit EventHandle(unsigned int flags = hipEventDisableTiming)
        {
            HIP_CHECK_EXC(hipEventCreateWithFlags(&event, flags));
        }

        ~EventHandle()
        {
            if(event)
                (void)hipEventDestroy(event);
        }

        EventHandle(EventHandle const&)            = delete;
        EventHandle& operator=(EventHandle const&) = delete;

        hipEvent_t get() const
        {
            return event;
        }
    };

    struct DeviceSignalBuffer
    {
        uint32_t* ptr = nullptr;

        ~DeviceSignalBuffer()
        {
            if(ptr)
                (void)hipFree(ptr);
        }

        void allocate()
        {
            void* raw = nullptr;
            HIP_CHECK_EXC(hipExtMallocWithFlags(&raw, sizeof(uint32_t), hipMallocSignalMemory));
            ptr = static_cast<uint32_t*>(raw);
        }

        uint32_t* get() const
        {
            return ptr;
        }
    };

    struct HostByteBuffer
    {
        uint8_t* ptr = nullptr;

        ~HostByteBuffer()
        {
            if(ptr)
                (void)hipHostFree(ptr);
        }

        void allocate()
        {
            void* raw = nullptr;
            HIP_CHECK_EXC(hipHostMalloc(&raw, sizeof(uint8_t), 0));
            ptr = static_cast<uint8_t*>(raw);
        }

        uint8_t* get() const
        {
            return ptr;
        }
    };

    ::testing::AssertionResult hasHipDeviceAndWaitValueSupport()
    {
        int        deviceCount = 0;
        hipError_t err         = hipGetDeviceCount(&deviceCount);
        if(err != hipSuccess)
        {
            return ::testing::AssertionFailure()
                   << "hipGetDeviceCount failed: " << hipGetErrorString(err);
        }
        if(deviceCount <= 0)
        {
            return ::testing::AssertionFailure() << "No HIP devices available";
        }

        int device = 0;
        err        = hipGetDevice(&device);
        if(err != hipSuccess)
            device = 0;

        int canUseStreamWaitValue = 0;
        err = hipDeviceGetAttribute(&canUseStreamWaitValue,
                                    hipDeviceAttributeCanUseStreamWaitValue,
                                    device);
        if(err != hipSuccess)
        {
            return ::testing::AssertionFailure()
                   << "hipDeviceGetAttribute(CanUseStreamWaitValue) failed: "
                   << hipGetErrorString(err);
        }
        if(!canUseStreamWaitValue)
        {
            return ::testing::AssertionFailure()
                   << "Device does not support hipStreamWaitValue32";
        }

        return ::testing::AssertionSuccess();
    }

    class NaNResetOutputDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using DataInitialization::resetOutput;

        std::vector<void*>& gpuPtrs()
        {
            return m_gpuPtrs;
        }

        std::vector<void**>& gpuBatchPtrs()
        {
            return m_gpuBatchPtrs;
        }

        std::vector<size_t>& maxElements()
        {
            return m_maxElements;
        }

        std::vector<std::vector<size_t>>& groupedOffsets()
        {
            return m_groupedOffsets;
        }

        PristineUnit& dPristineUnit(ContractionProblemGemm const& problem)
        {
            auto const tensorIndex = ContractionProblemGemm::TENSOR::D;
            auto const& desc       = problem.tensors().at(tensorIndex);
            auto&       units      = m_vdata.at(tensorIndex).pristine;
            auto        it         = units.find(desc.dataType());
            if(it == units.end())
            {
                throw std::runtime_error("Missing D pristine unit for problem data type.");
            }
            return it->second;
        }
    };

    uint8_t readFirstByte(void* devicePtr)
    {
        HostByteBuffer host;
        host.allocate();

        StreamHandle probeStream;
        HIP_CHECK_EXC(hipMemcpyAsync(host.get(),
                                     devicePtr,
                                     sizeof(uint8_t),
                                     hipMemcpyDeviceToHost,
                                     probeStream.get()));
        HIP_CHECK_EXC(hipStreamSynchronize(probeStream.get()));
        return *host.get();
    }
} // namespace

TEST(DataInitializationAsyncReset, NaNResetOutputD2DUsesTargetStream)
{
    auto hipDevice = hasHipDeviceAndWaitValueSupport();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);

    auto args = TensileLite::testing::buildBaseDataInitArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "bounds-check",
                                                 std::any(BoundsCheckMode::NaN));
    TensileLite::testing::detail::setDataInitArg(args, "pristine-on-gpu", std::any(true));
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "num-elements-to-validate",
                                                 std::any(int(1)));

    ClientProblemFactory             factory(args);
    NaNResetOutputDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    auto* ci = dynamic_cast<ContractionInputs*>(inputs.get());
    ASSERT_NE(ci, nullptr);
    ASSERT_NE(ci->d, nullptr);

    auto&       dUnit  = dataInit.dPristineUnit(problem);
    auto const& dDesc  = problem.tensors().at(ContractionProblemGemm::TENSOR::D);
    auto const   dSize = multiplyElementSize(dUnit.maxElements, dDesc.elementBytes());
    ASSERT_NE(dUnit.gpuInput.current, nullptr);
    ASSERT_NE(dUnit.gpuInput.bad, nullptr);

    HIP_CHECK_EXC(hipMemset(dUnit.gpuInput.current.get(), 0x00, dSize));
    HIP_CHECK_EXC(hipMemset(dUnit.gpuInput.bad.get(), 0xA5, dSize));

    DeviceSignalBuffer gateValue;
    gateValue.allocate();
    uint32_t zero = 0;
    HIP_CHECK_EXC(hipMemcpy(gateValue.get(), &zero, sizeof(zero), hipMemcpyHostToDevice));

    StreamHandle gateStream;
    StreamHandle targetStream;
    StreamHandle releaseStream;
    EventHandle  gateEvent;

    HIP_CHECK_EXC(hipStreamWaitValue32(
        gateStream.get(), gateValue.get(), 1, hipStreamWaitValueEq));
    HIP_CHECK_EXC(hipEventRecord(gateEvent.get(), gateStream.get()));
    HIP_CHECK_EXC(hipStreamWaitEvent(targetStream.get(), gateEvent.get(), 0));

    dataInit.resetOutput(dataInit.gpuPtrs(),
                         dataInit.gpuBatchPtrs(),
                         dataInit.maxElements(),
                         dataInit.groupedOffsets(),
                         problem,
                         hipMemcpyDeviceToDevice,
                         targetStream.get());

    EXPECT_EQ(hipEventQuery(gateEvent.get()), hipErrorNotReady);

    EXPECT_EQ(readFirstByte(dUnit.gpuInput.current.get()), 0x00)
        << "resetOutput should not run its D2D NaN reset before the target stream gate";

    HIP_CHECK_EXC(hipStreamWriteValue32(releaseStream.get(), gateValue.get(), 1, 0));
    HIP_CHECK_EXC(hipStreamSynchronize(targetStream.get()));

    EXPECT_EQ(readFirstByte(dUnit.gpuInput.current.get()), 0xA5)
        << "resetOutput should enqueue its D2D NaN reset on the caller stream";
}
