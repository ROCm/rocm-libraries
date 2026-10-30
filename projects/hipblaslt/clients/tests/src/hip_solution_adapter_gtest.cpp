// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <gtest/gtest.h>

#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

extern "C" void hipblaslt_debug_reload();

namespace
{
    constexpr char primary[] = "/library/primary.co";
    constexpr char helper[]  = "/library/Kernels.so-000-gfx942.hsaco";
    constexpr char minus[]   = "/library/Kernels.so-000-gfx942-xnack-.hsaco";
    constexpr char plus[]    = "/library/Kernels.so-000-gfx942-xnack+.hsaco";

    struct HipLoads
    {
        std::deque<std::pair<std::string, hipError_t>> expected;
        std::map<hipModule_t, std::unique_ptr<int>>    live;
        hipError_t                                     lastError = hipSuccess;
        unsigned                                       unloads   = 0;
    };

    // Each test owns this state and destroys the adapter before releasing it.
    HipLoads* hipLoads = nullptr;

    hipError_t moduleLoad(hipModule_t* module, const char* path);
    hipError_t moduleUnload(hipModule_t module);
    hipError_t getLastError();

    class HipSolutionAdapterTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            hipLoads = &loads;
            adapter  = std::make_unique<TensileLite::hip::SolutionAdapter>(
                TensileLite::hip::SolutionAdapter::ModuleApi{
                    moduleLoad, moduleUnload, getLastError});
        }

        void TearDown() override
        {
            adapter.reset();
            EXPECT_TRUE(loads.live.empty()) << "adapter leaked module handles";
            EXPECT_TRUE(loads.expected.empty()) << "expected module loads did not run";
            hipLoads = nullptr;
        }

        void expectLoad(const char* path, hipError_t error = hipSuccess)
        {
            loads.expected.emplace_back(path, error);
        }

        void expectHelperFailure(hipError_t error)
        {
            expectLoad(helper, error);
            expectLoad(minus, error);
            expectLoad(plus, error);
        }

        HipLoads                                           loads;
        std::unique_ptr<TensileLite::hip::SolutionAdapter> adapter;
    };

    class HipSolutionAdapterRecoveryTest : public HipSolutionAdapterTest,
                                           public ::testing::WithParamInterface<hipError_t>
    {
    };

    hipError_t moduleLoad(hipModule_t* module, const char* path)
    {
        // Unexpected recursive recovery fails the test and returns a non-retryable
        // error, rather than hanging the test process or exhausting its stack.
        if(hipLoads->expected.empty())
        {
            ADD_FAILURE() << "unexpected module load: " << path;
            return hipErrorInvalidValue;
        }
        auto [expectedPath, error] = hipLoads->expected.front();
        hipLoads->expected.pop_front();
        EXPECT_EQ(expectedPath, path);
        if(error == hipSuccess)
        {
            auto storage = std::make_unique<int>(0);
            *module      = reinterpret_cast<hipModule_t>(storage.get());
            hipLoads->live.emplace(*module, std::move(storage));
        }
        else
            hipLoads->lastError = error;
        return error;
    }

    hipError_t moduleUnload(hipModule_t module)
    {
        EXPECT_EQ(hipLoads->live.erase(module), 1u) << "double or unowned module unload";
        ++hipLoads->unloads;
        return hipSuccess;
    }

    hipError_t getLastError()
    {
        return std::exchange(hipLoads->lastError, hipSuccess);
    }

    TEST_F(HipSolutionAdapterTest,
           smoke_ContextDoesNotLoadHelpersAndInitializationDoesNotDuplicateThem)
    {
        adapter->setLazyLoadingContext("gfx942:sramecc+:xnack-", "/library");
        EXPECT_TRUE(loads.live.empty());
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942:sramecc+:xnack-", "/library"), hipSuccess);
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library/"), hipSuccess);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    TEST_F(HipSolutionAdapterTest, smoke_XnackFallbackRegistersHelperWithoutDuplicateLoads)
    {
        expectLoad(helper, hipErrorFileNotFound);
        expectLoad(minus);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942:sramecc+:xnack-", "/library"), hipSuccess);
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        EXPECT_EQ(loads.live.size(), 1u);
    }

    TEST_F(HipSolutionAdapterTest, smoke_OutOfMemoryDoesNotEnterRecovery)
    {
        adapter->setLazyLoadingContext("gfx942", "/library");
        expectLoad(primary, hipErrorOutOfMemory);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipErrorOutOfMemory);
        EXPECT_EQ(loads.unloads, 0u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_MissingContextPreservesPrimaryError)
    {
        expectLoad(primary, GetParam());
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), GetParam());
        EXPECT_EQ(loads.unloads, 0u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest,
           smoke_EarlyContextAllowsRecoveryBeforeHelperInitialization)
    {
        adapter->setLazyLoadingContext("gfx942:sramecc+:xnack-", "/library");
        expectLoad(primary, GetParam());
        expectLoad(helper);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_RecoveryUnloadsOldModulesAndRestoresHelpers)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary, GetParam());
        expectLoad(helper);
        expectLoad(primary);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        EXPECT_EQ(loads.unloads, 1u);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_MissingHelpersPreservePrimaryError)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary, GetParam());
        expectHelperFailure(hipErrorFileNotFound);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), GetParam());
        EXPECT_EQ(loads.unloads, 1u);
        EXPECT_TRUE(loads.live.empty());
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_RecoverableHelperErrorsDoNotRecurse)
    {
        adapter->setLazyLoadingContext("gfx942", "/library");
        expectLoad(primary, hipErrorLaunchFailure);
        expectHelperFailure(GetParam());
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipErrorLaunchFailure);
        EXPECT_TRUE(loads.live.empty());
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_DirectHelperInitializationDoesNotRecurse)
    {
        expectHelperFailure(GetParam());
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), GetParam());
        EXPECT_EQ(loads.unloads, 0u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_FailedPrimaryRetryIsReturnedWithoutAnotherRecovery)
    {
        adapter->setLazyLoadingContext("gfx942", "/library");
        expectLoad(primary, hipErrorLaunchFailure);
        expectLoad(helper);
        expectLoad(primary, GetParam());
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), GetParam());
        EXPECT_EQ(loads.live.size(), 1u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_RecoveryReclaimsRotationCopies)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        expectLoad(primary);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFileExtraCopies(primary, 2), hipSuccess);
        adapter->selectRotationCopy(2);
        ASSERT_EQ(adapter->numRotationModules(), 3);
        expectLoad(primary, GetParam());
        expectLoad(helper);
        expectLoad(primary);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        EXPECT_EQ(adapter->numRotationModules(), 1);
        EXPECT_EQ(loads.unloads, 4u);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    INSTANTIATE_TEST_SUITE_P(RecoverableErrors,
                             HipSolutionAdapterRecoveryTest,
                             ::testing::Values(hipErrorLaunchFailure, hipErrorNoBinaryForGpu));

    // ----------------------------------------------------------------------
    // Whose failure an unrecordable launch event is.
    //
    // The events launchKernel takes are instrumentation, but they are recorded
    // around a production dispatch, and HIP_CHECK_RETURN made hipEventRecord's
    // error the launch's error. Online tuning times the caller's own
    // hipblasLtMatmul with them, so that turned a measurement problem into a
    // failed GEMM -- and a start record that fails returns before the kernel
    // is enqueued, so the GEMM does not run either.
    //
    // bestEffortEvents is the caller saying it would rather lose the
    // measurement. The two cases below are the same launch with that answer
    // given both ways.
    //
    // Building an unrecordable event takes two devices, so the suite carries
    // the multi_gpu token and no smoke_ prefix: CI runs it in its multi_gpu
    // tier rather than selecting it in every single-GPU tier, where it could
    // only skip. See clients/tests/test_categories.yaml.
    // ----------------------------------------------------------------------

    class HipSolutionAdapterEventTest_multi_gpu : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(hipGetDeviceCount(&m_devices) != hipSuccess || m_devices < 2)
                GTEST_SKIP() << "needs two HIP devices to build an unrecordable event";

            // The launch has to reach the event records without a module or a
            // kernel, so the records are what is under test rather than
            // whatever fails first. TENSILE_DB2 bit 0 is skip-kernel-launch,
            // and the adapter caches it at construction, hence the local one.
            // If the reload below ever stops reaching the adapter these tests
            // fail with hipErrorNotFound from getKernel, not a wrong answer.
            //
            // Reloaded through the library hook, not Debug::Instance() here.
            // Debug is a LazySingleton, a function-local static in a header,
            // so this binary's copy is not the one the adapter reads -- the
            // adapter lives in libtensilelite-host and has a third.
            const char* const existing = getenv("TENSILE_DB2");
            m_savedDb2 = existing ? std::optional<std::string>(existing) : std::nullopt;
            setenv("TENSILE_DB2", "0x1", 1);
            hipblaslt_debug_reload();

            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            ASSERT_EQ(hipEventCreateWithFlags(&m_onDeviceZero, hipEventDefault), hipSuccess);

            // Recording an event on another device's stream is rejected with
            // hipErrorInvalidResourceHandle -- measured on gfx950, and the
            // failure a process driving several GPUs actually hits. A
            // destroyed handle would be the other way to get a failing record
            // and it faults instead of returning.
            ASSERT_EQ(hipSetDevice(1), hipSuccess);
            ASSERT_EQ(hipStreamCreate(&m_streamOnDeviceOne), hipSuccess);
            ASSERT_NE(hipEventRecord(m_onDeviceZero, m_streamOnDeviceOne), hipSuccess)
                << "this device pair records across devices, so nothing here can fail";
            static_cast<void>(hipGetLastError());
        }

        void TearDown() override
        {
            if(m_streamOnDeviceOne)
                static_cast<void>(hipStreamDestroy(m_streamOnDeviceOne));
            if(m_onDeviceZero)
            {
                static_cast<void>(hipSetDevice(0));
                static_cast<void>(hipEventDestroy(m_onDeviceZero));
            }

            if(m_savedDb2)
                setenv("TENSILE_DB2", m_savedDb2->c_str(), 1);
            else
                unsetenv("TENSILE_DB2");
            hipblaslt_debug_reload();

            static_cast<void>(hipSetDevice(0));
        }

        // One kernel with no code object, so the launch is all event handling.
        std::vector<TensileLite::KernelInvocation> kernels() const
        {
            return std::vector<TensileLite::KernelInvocation>(1);
        }

        int                        m_devices           = 0;
        hipEvent_t                 m_onDeviceZero      = nullptr;
        hipStream_t                m_streamOnDeviceOne = nullptr;
        std::optional<std::string> m_savedDb2;
    };

    TEST_F(HipSolutionAdapterEventTest_multi_gpu, AnUnrecordableEventFailsTheLaunchByDefault)
    {
        TensileLite::hip::SolutionAdapter adapter;

        EXPECT_EQ(adapter.launchKernels(kernels(), m_streamOnDeviceOne, m_onDeviceZero, nullptr),
                  hipErrorInvalidResourceHandle);
        EXPECT_EQ(adapter.launchKernels(kernels(), m_streamOnDeviceOne, nullptr, m_onDeviceZero),
                  hipErrorInvalidResourceHandle);
        static_cast<void>(hipGetLastError());
    }

    TEST_F(HipSolutionAdapterEventTest_multi_gpu, BestEffortEventsCostTheMeasurementNotTheLaunch)
    {
        TensileLite::hip::SolutionAdapter adapter;

        EXPECT_EQ(adapter.launchKernels(kernels(),
                                        m_streamOnDeviceOne,
                                        m_onDeviceZero,
                                        m_onDeviceZero,
                                        /*isKernelLoaded=*/false,
                                        /*bestEffortEvents=*/true),
                  hipSuccess)
            << "a measurement event failure was returned as the launch's status";

        // And it is not left behind for the caller to trip over later.
        EXPECT_EQ(hipGetLastError(), hipSuccess)
            << "the swallowed event error was left in HIP's last-error state";
    }

    // The other half of not leaving an error behind: an error the caller
    // already had, and has not read yet, is the caller's. Swallowing a failed
    // record must not clear it along with the record's own.
    TEST_F(HipSolutionAdapterEventTest_multi_gpu, BestEffortEventsLeaveTheCallersErrorInPlace)
    {
        TensileLite::hip::SolutionAdapter adapter;

        ASSERT_NE(hipSetDevice(m_devices), hipSuccess) << "an out-of-range device was accepted";
        ASSERT_NE(hipPeekAtLastError(), hipSuccess) << "the caller's error did not stick";

        EXPECT_EQ(adapter.launchKernels(kernels(),
                                        m_streamOnDeviceOne,
                                        m_onDeviceZero,
                                        m_onDeviceZero,
                                        /*isKernelLoaded=*/false,
                                        /*bestEffortEvents=*/true),
                  hipSuccess);

        EXPECT_NE(hipGetLastError(), hipSuccess)
            << "a swallowed event error cleared an error the caller had not read";
    }

    // A swallowed start record must not leave a recorded stop behind it.
    //
    // Online tuning reads a pair back on a later visit and keeps it only if the
    // elapsed time is positive, relying on every unrecorded pair reading
    // otherwise. A half-recorded pair breaks that when the pair is reused: its
    // start still holds the timestamp of an older measurement, so a fresh stop
    // against it reads as the time since then, a plausible positive duration.
    // So when the start record is swallowed the stop is skipped too, and the
    // pair reads as never recorded. Asserted for one kernel, where both records
    // are in one launchKernel, and for a solution of several, where the start
    // goes with the first kernel and the stop with the last.
    TEST_F(HipSolutionAdapterEventTest_multi_gpu, ASwallowedStartRecordLeavesTheStopUnrecorded)
    {
        TensileLite::hip::SolutionAdapter adapter;

        for(size_t count : {size_t(1), size_t(3)})
        {
            // On device one, current since SetUp, so it records on the stream
            // the launch uses: only the start can fail.
            hipEvent_t stop = nullptr;
            ASSERT_EQ(hipEventCreateWithFlags(&stop, hipEventDefault), hipSuccess);

            EXPECT_EQ(adapter.launchKernels(std::vector<TensileLite::KernelInvocation>(count),
                                            m_streamOnDeviceOne,
                                            m_onDeviceZero,
                                            stop,
                                            /*isKernelLoaded=*/false,
                                            /*bestEffortEvents=*/true),
                      hipSuccess)
                << count << " kernels";
            EXPECT_EQ(hipStreamSynchronize(m_streamOnDeviceOne), hipSuccess);

            // An event never recorded has no timestamp to read.
            float milliseconds = 0.0f;
            EXPECT_NE(hipEventElapsedTime(&milliseconds, stop, stop), hipSuccess)
                << "with " << count << " kernels the stop was recorded although the start was not";
            static_cast<void>(hipGetLastError());

            static_cast<void>(hipEventDestroy(stop));
        }
    }

} // namespace
